//! GPU hash kernels (Stage A) — real CUDA via cudarc.
//!
//! Loads the PTX compiled by `build.rs` and runs SHA-256 / HASH160 on-device.
//! This is the toolchain proof and the hashing half of the eventual BIP32
//! derivation kernel; correctness is checked against the same vectors the CPU
//! path uses (see tests and `address::hash160`).
//!
//! Only built with `--features cuda`.

use crate::cuda::CudaError;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// Max single-block message length (SHA-256/RIPEMD-160 one-block limit).
pub const MAX_MSG_LEN: usize = 55;

const PTX: &str = include_str!(env!("KERNEL_HASH160_PTX"));

/// Number of bytes in the fixed-base table: 32 windows × 255 entries × 64 bytes.
const GTABLE_BYTES: usize = 32 * 255 * 64;

/// Build the windowed fixed-base table for `k·G` (Stage E).
///
/// Layout: entry `(w, d)` = `d * 256^w * G` as affine `X||Y` big-endian
/// (64 bytes), for `w` in `0..32`, `d` in `1..=255`. Generated once via k256.
fn build_gtable() -> Vec<u8> {
    use k256::elliptic_curve::sec1::ToEncodedPoint;
    use k256::ProjectivePoint;

    let mut table = vec![0u8; GTABLE_BYTES];
    let mut window_base = ProjectivePoint::GENERATOR; // 256^0 * G
    for w in 0..32 {
        let mut acc = window_base; // d = 1
        for d in 1..=255usize {
            let enc = acc.to_affine().to_encoded_point(false);
            let off = (w * 255 + (d - 1)) * 64;
            table[off..off + 32].copy_from_slice(&enc.x().unwrap()[..]);
            table[off + 32..off + 64].copy_from_slice(&enc.y().unwrap()[..]);
            acc += window_base;
        }
        for _ in 0..8 {
            window_base = window_base.double(); // *= 256
        }
    }
    table
}

/// A loaded hash-kernel module bound to one device/stream.
pub struct HashKernels {
    ctx: Arc<CudaContext>,
    stream: Arc<cudarc::driver::CudaStream>,
    sha256: cudarc::driver::CudaFunction,
    hash160: cudarc::driver::CudaFunction,
}

impl HashKernels {
    pub fn new(device_id: usize) -> Result<Self, CudaError> {
        let ctx = CudaContext::new(device_id)
            .map_err(|e| CudaError::DriverError(format!("context: {e}")))?;
        let module = ctx
            .load_module(Ptx::from_src(PTX))
            .map_err(|e| CudaError::KernelError(format!("load ptx: {e}")))?;
        let sha256 = module
            .load_function("sha256_kernel")
            .map_err(|e| CudaError::KernelError(format!("sha256_kernel: {e}")))?;
        let hash160 = module
            .load_function("hash160_kernel")
            .map_err(|e| CudaError::KernelError(format!("hash160_kernel: {e}")))?;
        let stream = ctx.default_stream();
        Ok(Self { ctx, stream, sha256, hash160 })
    }

    /// Pack messages into a flat `n * MAX_MSG_LEN` buffer + lengths array.
    fn pack(msgs: &[&[u8]]) -> Result<(Vec<u8>, Vec<u32>), CudaError> {
        let n = msgs.len();
        let mut flat = vec![0u8; n * MAX_MSG_LEN];
        let mut lens = vec![0u32; n];
        for (i, m) in msgs.iter().enumerate() {
            if m.len() > MAX_MSG_LEN {
                return Err(CudaError::KernelError(format!(
                    "message {i} len {} exceeds single-block max {MAX_MSG_LEN}",
                    m.len()
                )));
            }
            flat[i * MAX_MSG_LEN..i * MAX_MSG_LEN + m.len()].copy_from_slice(m);
            lens[i] = m.len() as u32;
        }
        Ok((flat, lens))
    }

    fn run(
        &self,
        func: &cudarc::driver::CudaFunction,
        msgs: &[&[u8]],
        out_elem: usize,
    ) -> Result<Vec<u8>, CudaError> {
        let n = msgs.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let (flat, lens) = Self::pack(msgs)?;

        let d_msgs = self
            .stream
            .clone_htod(&flat)
            .map_err(|e| CudaError::MemoryError(format!("htod msgs: {e}")))?;
        let d_lens = self
            .stream
            .clone_htod(&lens)
            .map_err(|e| CudaError::MemoryError(format!("htod lens: {e}")))?;
        let mut d_out = self
            .stream
            .alloc_zeros::<u8>(n * out_elem)
            .map_err(|e| CudaError::MemoryError(format!("alloc out: {e}")))?;

        let stride = MAX_MSG_LEN as u32;
        let count = n as u32;
        let cfg = LaunchConfig::for_num_elems(count);
        let mut builder = self.stream.launch_builder(func);
        builder.arg(&d_msgs).arg(&d_lens).arg(&stride).arg(&count).arg(&mut d_out);
        unsafe { builder.launch(cfg) }
            .map_err(|e| CudaError::KernelError(format!("launch: {e}")))?;

        let mut host_out = vec![0u8; n * out_elem];
        self.stream
            .memcpy_dtoh(&d_out, &mut host_out)
            .map_err(|e| CudaError::MemoryError(format!("dtoh out: {e}")))?;
        self.stream
            .synchronize()
            .map_err(|e| CudaError::KernelError(format!("sync: {e}")))?;
        Ok(host_out)
    }

    /// SHA-256 each message (each <= 55 bytes). Returns `n * 32` bytes.
    pub fn sha256(&self, msgs: &[&[u8]]) -> Result<Vec<[u8; 32]>, CudaError> {
        let flat = self.run(&self.sha256.clone(), msgs, 32)?;
        Ok(flat.chunks_exact(32).map(|c| c.try_into().unwrap()).collect())
    }

    /// HASH160 = RIPEMD160(SHA256(x)) each message. Returns `n * 20` bytes.
    pub fn hash160(&self, msgs: &[&[u8]]) -> Result<Vec<[u8; 20]>, CudaError> {
        let flat = self.run(&self.hash160.clone(), msgs, 20)?;
        Ok(flat.chunks_exact(20).map(|c| c.try_into().unwrap()).collect())
    }

    /// Device name, for diagnostics.
    pub fn device_name(&self) -> String {
        self.ctx.name().unwrap_or_else(|_| "unknown".to_string())
    }
}

const PTX_SECP: &str = include_str!(env!("KERNEL_SECP256K1_PTX"));

/// secp256k1 `k·G` (fixed-base scalar multiply) on the GPU — Stage B.
pub struct EcKernels {
    stream: Arc<cudarc::driver::CudaStream>,
    scalar_mul_g: cudarc::driver::CudaFunction,
    d_gtable: cudarc::driver::CudaSlice<u8>,
}

impl EcKernels {
    pub fn new(device_id: usize) -> Result<Self, CudaError> {
        let ctx = CudaContext::new(device_id)
            .map_err(|e| CudaError::DriverError(format!("context: {e}")))?;
        let module = ctx
            .load_module(Ptx::from_src(PTX_SECP))
            .map_err(|e| CudaError::KernelError(format!("load secp ptx: {e}")))?;
        let scalar_mul_g = module
            .load_function("scalar_mul_g_kernel")
            .map_err(|e| CudaError::KernelError(format!("scalar_mul_g_kernel: {e}")))?;
        let stream = ctx.default_stream();
        let d_gtable = stream
            .clone_htod(&build_gtable())
            .map_err(|e| CudaError::MemoryError(format!("htod gtable: {e}")))?;
        Ok(Self { stream, scalar_mul_g, d_gtable })
    }

    /// Compute `k·G` for each 32-byte big-endian scalar; returns 33-byte
    /// compressed pubkeys.
    pub fn scalar_mul_g(&self, scalars: &[[u8; 32]]) -> Result<Vec<[u8; 33]>, CudaError> {
        let n = scalars.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut flat = vec![0u8; n * 32];
        for (i, s) in scalars.iter().enumerate() {
            flat[i * 32..i * 32 + 32].copy_from_slice(s);
        }
        let d_scalars = self
            .stream
            .clone_htod(&flat)
            .map_err(|e| CudaError::MemoryError(format!("htod scalars: {e}")))?;
        let mut d_out = self
            .stream
            .alloc_zeros::<u8>(n * 33)
            .map_err(|e| CudaError::MemoryError(format!("alloc out: {e}")))?;

        let count = n as u32;
        // This kernel is register-heavy (deep double-and-add, many fe temps),
        // so a 1024-thread block overflows the register file. Use 64.
        const BLOCK: u32 = 64;
        let cfg = LaunchConfig {
            grid_dim: (count.div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = self.stream.launch_builder(&self.scalar_mul_g);
        builder.arg(&d_scalars).arg(&self.d_gtable).arg(&count).arg(&mut d_out);
        unsafe { builder.launch(cfg) }
            .map_err(|e| CudaError::KernelError(format!("launch: {e}")))?;

        let mut host_out = vec![0u8; n * 33];
        self.stream
            .memcpy_dtoh(&d_out, &mut host_out)
            .map_err(|e| CudaError::MemoryError(format!("dtoh out: {e}")))?;
        self.stream
            .synchronize()
            .map_err(|e| CudaError::KernelError(format!("sync: {e}")))?;
        Ok(host_out.chunks_exact(33).map(|c| c.try_into().unwrap()).collect())
    }
}

const PTX_DERIVE: &str = include_str!(env!("KERNEL_DERIVE_PTX"));

/// One derived child: compressed pubkey + hash160 + validity status.
#[derive(Debug, Clone)]
pub struct GpuDerived {
    pub index: u32,
    pub pubkey: [u8; 33],
    pub hash160: [u8; 20],
    /// 0 = valid; 1 = BIP32-invalid (IL ≥ n or point at infinity).
    pub status: u8,
}

/// BIP32 non-hardened public-side batch derivation on the GPU — Stage C.
pub struct DeriveKernel {
    stream: Arc<cudarc::driver::CudaStream>,
    func: cudarc::driver::CudaFunction,
    d_gtable: cudarc::driver::CudaSlice<u8>,
}

impl DeriveKernel {
    pub fn new(device_id: usize) -> Result<Self, CudaError> {
        let ctx = CudaContext::new(device_id)
            .map_err(|e| CudaError::DriverError(format!("context: {e}")))?;
        let module = ctx
            .load_module(Ptx::from_src(PTX_DERIVE))
            .map_err(|e| CudaError::KernelError(format!("load derive ptx: {e}")))?;
        let func = module
            .load_function("derive_child_kernel")
            .map_err(|e| CudaError::KernelError(format!("derive_child_kernel: {e}")))?;
        let stream = ctx.default_stream();
        let d_gtable = stream
            .clone_htod(&build_gtable())
            .map_err(|e| CudaError::MemoryError(format!("htod gtable: {e}")))?;
        Ok(Self { stream, func, d_gtable })
    }

    /// Decompress a compressed pubkey to affine (x, y) big-endian via k256.
    fn decompress(pubkey: &[u8; 33]) -> Result<([u8; 32], [u8; 32]), CudaError> {
        use k256::elliptic_curve::sec1::{FromEncodedPoint, ToEncodedPoint};
        use k256::{AffinePoint, EncodedPoint};
        let ep = EncodedPoint::from_bytes(pubkey)
            .map_err(|e| CudaError::KernelError(format!("bad pubkey: {e}")))?;
        let ap = Option::<AffinePoint>::from(AffinePoint::from_encoded_point(&ep))
            .ok_or_else(|| CudaError::KernelError("pubkey not on curve".into()))?;
        let unc = ap.to_encoded_point(false);
        let x: [u8; 32] = (&unc.x().unwrap()[..]).try_into().unwrap();
        let y: [u8; 32] = (&unc.y().unwrap()[..]).try_into().unwrap();
        Ok((x, y))
    }

    /// Derive `count` non-hardened children starting at `start_index`.
    pub fn derive_batch(
        &self,
        pubkey: &[u8; 33],
        chain_code: &[u8; 32],
        start_index: u32,
        count: u32,
    ) -> Result<Vec<GpuDerived>, CudaError> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let n = count as usize;
        let (px, py) = Self::decompress(pubkey)?;

        let d_pub = self.stream.clone_htod(pubkey.as_slice())
            .map_err(|e| CudaError::MemoryError(format!("htod pub: {e}")))?;
        let d_cc = self.stream.clone_htod(chain_code.as_slice())
            .map_err(|e| CudaError::MemoryError(format!("htod cc: {e}")))?;
        let d_px = self.stream.clone_htod(px.as_slice())
            .map_err(|e| CudaError::MemoryError(format!("htod px: {e}")))?;
        let d_py = self.stream.clone_htod(py.as_slice())
            .map_err(|e| CudaError::MemoryError(format!("htod py: {e}")))?;
        let mut d_out_pub = self.stream.alloc_zeros::<u8>(n * 33)
            .map_err(|e| CudaError::MemoryError(format!("alloc pub: {e}")))?;
        let mut d_out_h160 = self.stream.alloc_zeros::<u8>(n * 20)
            .map_err(|e| CudaError::MemoryError(format!("alloc h160: {e}")))?;
        let mut d_out_status = self.stream.alloc_zeros::<u8>(n)
            .map_err(|e| CudaError::MemoryError(format!("alloc status: {e}")))?;

        // Resource-heavy kernel (HMAC-SHA512 + EC) with a large per-thread
        // local-memory footprint; 64 measured best on sm_86 (128 lowered
        // occupancy and throughput).
        const BLOCK: u32 = 64;
        let cfg = LaunchConfig {
            grid_dim: (count.div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(&self.func);
        b.arg(&d_pub).arg(&d_cc).arg(&d_px).arg(&d_py).arg(&self.d_gtable)
            .arg(&start_index).arg(&count)
            .arg(&mut d_out_pub).arg(&mut d_out_h160).arg(&mut d_out_status);
        unsafe { b.launch(cfg) }
            .map_err(|e| CudaError::KernelError(format!("launch: {e}")))?;

        let mut h_pub = vec![0u8; n * 33];
        let mut h_h160 = vec![0u8; n * 20];
        let mut h_status = vec![0u8; n];
        self.stream.memcpy_dtoh(&d_out_pub, &mut h_pub)
            .map_err(|e| CudaError::MemoryError(format!("dtoh pub: {e}")))?;
        self.stream.memcpy_dtoh(&d_out_h160, &mut h_h160)
            .map_err(|e| CudaError::MemoryError(format!("dtoh h160: {e}")))?;
        self.stream.memcpy_dtoh(&d_out_status, &mut h_status)
            .map_err(|e| CudaError::MemoryError(format!("dtoh status: {e}")))?;
        self.stream.synchronize()
            .map_err(|e| CudaError::KernelError(format!("sync: {e}")))?;

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(GpuDerived {
                index: start_index + i as u32,
                pubkey: h_pub[i * 33..i * 33 + 33].try_into().unwrap(),
                hash160: h_h160[i * 20..i * 20 + 20].try_into().unwrap(),
                status: h_status[i],
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These tests need a real GPU; skip gracefully if none is present.
    fn kernels() -> Option<HashKernels> {
        HashKernels::new(0).ok()
    }

    #[test]
    fn sha256_abc_vector() {
        let Some(k) = kernels() else {
            eprintln!("no CUDA device; skipping");
            return;
        };
        let out = k.sha256(&[b"abc"]).unwrap();
        let expected =
            hex::decode("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
                .unwrap();
        assert_eq!(out[0].to_vec(), expected);
    }

    #[test]
    fn ripemd_sha_hash160_generator_vector() {
        let Some(k) = kernels() else {
            eprintln!("no CUDA device; skipping");
            return;
        };
        // Same vector as address.rs:88 — hash160 of the compressed generator pubkey.
        let pubkey =
            hex::decode("0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798")
                .unwrap();
        let out = k.hash160(&[&pubkey]).unwrap();
        let expected = hex::decode("751e76e8199196d454941c45d1b3a323f1433bd6").unwrap();
        assert_eq!(out[0].to_vec(), expected);
    }

    #[test]
    fn scalar_mul_g_matches_k256() {
        use k256::elliptic_curve::ops::Reduce;
        use k256::elliptic_curve::sec1::ToEncodedPoint;
        use k256::{ProjectivePoint, Scalar, U256};

        let Some(()) = EcKernels::new(0).ok().map(|_| ()) else {
            eprintln!("no CUDA device; skipping");
            return;
        };
        let ec = EcKernels::new(0).unwrap();

        // Deterministic spread of scalars: 1, 2, a few fixed, then an LCG.
        let mut scalars: Vec<[u8; 32]> = Vec::new();
        let mut push_u64 = |v: u64| {
            let mut s = [0u8; 32];
            s[24..32].copy_from_slice(&v.to_be_bytes());
            scalars.push(s);
        };
        push_u64(1);
        push_u64(2);
        push_u64(3);
        push_u64(0xDEAD_BEEF);
        // LCG-generated 256-bit-ish scalars (fill all 32 bytes).
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        for _ in 0..60 {
            let mut s = [0u8; 32];
            for chunk in s.chunks_mut(8) {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                chunk.copy_from_slice(&state.to_be_bytes());
            }
            scalars.push(s);
        }

        // Reduce each scalar mod n so the value matches what we feed the GPU.
        let reduced: Vec<[u8; 32]> = scalars
            .iter()
            .map(|s| {
                let k = Scalar::reduce(U256::from_be_slice(s));
                k.to_bytes().into()
            })
            .collect();

        let gpu = ec.scalar_mul_g(&reduced).unwrap();

        for (i, s) in reduced.iter().enumerate() {
            let k = Scalar::reduce(U256::from_be_slice(s));
            let expected = ProjectivePoint::GENERATOR * k;
            let enc = expected.to_affine().to_encoded_point(true);
            assert_eq!(&gpu[i][..], enc.as_bytes(), "mismatch at scalar {i}");
        }
    }

    #[test]
    fn derive_batch_matches_cpu() {
        use crate::address::hash160;
        use crate::bip32::ExtendedPrivateKey;

        let Some(_) = DeriveKernel::new(0).ok() else {
            eprintln!("no CUDA device; skipping");
            return;
        };
        let dk = DeriveKernel::new(0).unwrap();

        // Same canonical xpub as batch.rs tests: seed=[0;64], m/84'/0'/0'.
        let master = ExtendedPrivateKey::from_seed(&[0u8; 64]).unwrap();
        let xpub = master.derive_path("m/84'/0'/0'").unwrap().public_key();

        let count = 512u32;
        let gpu = dk.derive_batch(&xpub.key, &xpub.chain_code, 0, count).unwrap();
        assert_eq!(gpu.len() as u32, count);

        for g in &gpu {
            let cpu = xpub.derive_child(g.index).expect("cpu derive");
            assert_eq!(g.status, 0, "index {} unexpectedly invalid", g.index);
            assert_eq!(g.pubkey, cpu.key, "pubkey mismatch at {}", g.index);
            assert_eq!(g.hash160, hash160(&cpu.key), "hash160 mismatch at {}", g.index);
        }
    }

    #[test]
    fn hash160_matches_cpu_batch() {
        let Some(k) = kernels() else {
            eprintln!("no CUDA device; skipping");
            return;
        };
        // A batch of distinct 33-byte inputs; GPU hash160 must match CPU hash160.
        let inputs: Vec<Vec<u8>> = (0u8..64)
            .map(|i| {
                let mut v = vec![0x02u8];
                v.extend_from_slice(&[i; 32]);
                v
            })
            .collect();
        let refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
        let gpu = k.hash160(&refs).unwrap();
        for (i, inp) in inputs.iter().enumerate() {
            let cpu = crate::address::hash160(inp);
            assert_eq!(gpu[i], cpu, "mismatch at {i}");
        }
    }
}
