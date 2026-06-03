# Implementation Plan — Real CUDA Batch Derivation

> Scope of THIS plan: replace the fake "GPU" path (CPU compute + pointless
> host↔device copies in `cuda.rs:212-234`, and CPU + fabricated timings in
> `cuda.rs:351-401`) with a genuine GPU kernel that derives child public keys
> and HASH160 on-device. Phases 1/2/4 of the original project plan (BIP39,
> BIP32, CPU batch baseline, CLI) are already complete and are the reference
> oracle for correctness here.

## Goal

For a non-hardened xpub `(pubkey[33], chain_code[32])` and an index range
`[start, start+count)`, compute on the GPU — with **no per-index host
compute** — for every index `i`:

```
child_pubkey[i] (33 bytes)  ==  ExtendedPublicKey::derive_child(i).key
hash160[i]      (20 bytes)  ==  address::hash160(child_pubkey[i])
```

The host transfers only the constant xpub material + the index range; the GPU
writes back `count * (33 + 20)` bytes. Success = byte-for-byte match against the
existing CPU `derive_child` over the whole batch.

## Why this is tractable here (key design decisions)

These collapse "port all of secp256k1 to CUDA" into a much smaller kernel:

1. **child = IL·G + parent_point.** `G` is the fixed generator → use a
   **precomputed fixed-base table** (multiples of G), uploaded once. No
   variable-base scalar multiplication (the hard, slow part) is needed.
2. **`parent_point` is constant across the batch.** Decompress the xpub's
   pubkey to affine **once on the host**, upload `(X, Y)`. Each thread does one
   point addition against this constant.
3. **HMAC key = `chain_code` is a batch constant and < 128 B.** Precompute the
   SHA-512 states of `i_key_pad` and `o_key_pad` on the host and upload them.
   Per-thread HMAC then costs ~1 inner + 1 outer SHA-512 compression over the
   37-byte message `pubkey(33) || index_be(4)`.
4. **secp256k1 prime `p = 2^256 − 2^32 − 977`** has cheap reduction (no
   Montgomery domain required). Curve order `n` only needed for the `IL ≥ n`
   validity check (already in `bip32.rs:24` as `SECP256K1_ORDER`).

Per-thread work therefore reduces to: 2 SHA-512 compressions → reduce `IL` →
fixed-base `IL·G` → one point-add with a constant → compress point → SHA-256 →
RIPEMD-160.

## Edge cases (must match CPU semantics exactly)

- `IL ≥ n` or `IL == 0` → BIP32 says skip; CPU returns `Err`. Kernel must mark
  the slot invalid (e.g. write a sentinel / per-slot status byte) rather than
  emit a wrong key. Probability ~2^-128, but the harness must agree on it.
- `child_point == identity` → invalid (`Bip32Error::PointAtInfinity`).
- Point compression parity byte: `0x02`/`0x03` from `Y` LSB, matching
  `to_encoded_point(true)`.
- All scalars/coordinates are **big-endian** on the wire (matches
  `U256::from_be_slice` and `to_be_bytes`).

---

## Stage A: Build pipeline + GPU hash kernels
**Goal**: `nvcc → PTX → cudarc load → launch` works end-to-end, and SHA-256 /
RIPEMD-160 / SHA-512 kernels are correct in isolation.
**Why first**: hashing is self-contained, has abundant published test vectors,
and de-risks the whole toolchain before touching elliptic-curve math.

**Tasks**:
- Add `build.rs` that compiles `kernels/derive.cu` to PTX with `nvcc`
  (gated on `feature = "cuda"`); emit PTX path via `cargo:rustc-env` or
  `OUT_DIR`. Decide arch (`-arch=sm_80`/`sm_89`; make overridable via env).
- Reuse device hash functions rather than writing them: `sha256`/`ripemd160`/
  `hash160` from brainsecp `GPU/GPUHash.h` (MIT); `sha512` from recovery
  (Apache-2.0). See "Reuse map".
- A throwaway `test_hash` kernel that hashes host-provided inputs back to host.

**Tests / Success Criteria**:
- `hash160(0279BE66…F81798) == 751e76e8…33bd6` (same vector as
  `address.rs:88`).  ✅ passing on RTX 3070
- SHA-256 of `"abc"` matches NIST vector.  ✅ passing
- Batch of 64 distinct inputs: GPU `hash160` == CPU `address::hash160`.  ✅
- Launch + copyback verified through cudarc (not just compiled).  ✅

**Status**: COMPLETE (SHA-256 + RIPEMD-160 + hash160). Implemented as
`kernels/hash160.cu` + `build.rs` (nvcc→PTX, `sm_86`, env-overridable) +
`src/gpu_kernel.rs` (`HashKernels`, cudarc load/launch). Env: nvcc 12.8 + RTX
3070 Laptop present. SHA-512 deferred to Stage C (only needed for HMAC there).
Note: single-block hashing (msg ≤ 55 B) — sufficient for 33-B pubkeys; revisit
if multi-block is ever needed.

## Stage B: secp256k1 field + group arithmetic on GPU
**Goal**: Correct `Fp` arithmetic and point ops; fixed-base `k·G` and point
addition validated against known answers.
**Approach: ADAPT existing permissively-licensed kernels, do NOT write from
scratch.** A repo survey (see "Reuse map" below) found every primitive already
implemented. brainsecp's `_PointMultiSecp256k1` is, modulo the input scalar, the
exact fixed-base `k·G` needed here.
**Tasks**:
- Vendor brainsecp (MIT) `GPU/GPUMath.h` (field ops + `_PointAddSecp256k1`,
  mixed Jacobian+affine) and its host-side generator-table builder
  (`CPU/Point.cpp` / `SECP256K1.cpp`). Adapt `_PointMultiSecp256k1` so the
  scalar input is `IL` instead of a brain-wallet privkey.
- Confirm endianness/limb layout (it uses 4×u64 little-endian limbs) and bridge
  to the big-endian wire format used by the CPU oracle.
- Keep the compressed-`(parity,X)` path: brainsecp's `_GetHash160Comp` already
  takes `(qx, y&1)`.
- Alternative source if brainsecp's 16-bit-window table (~128 MB VRAM) is too
  heavy: recovery's `third_party/secp256k1/GPUGroup.cuh` ships a `G,2G,3G,…`
  generator table with a different window strategy.

**Tests / Success Criteria**:
- `1·G == 0279BE66…F81798` (compressed generator).  ✅
- `k·G` vs the `k256` crate over 64 scalars incl. full-256-bit values.  ✅

**Status**: COMPLETE. Chose the boring/verifiable path over vendoring: clean
`kernels/secp256k1.cu` — 8×u32 field arithmetic mod p (schoolbook mul + fold
reduce via `2^256 ≡ 2^32+977`), Jacobian `dbl-2009-l` / `add-2007-bl`, Fermat
inverse, MSB-first double-and-add `k·G`, compressed output. Host:
`EcKernels::scalar_mul_g` in `src/gpu_kernel.rs`, validated byte-for-byte vs
`k256`. NOTE: register-heavy kernel → launch with 64-thread blocks (1024
overflowed the register file: `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`). The
windowed fixed-base table (CudaBrainSecp) remains the **Stage E** speedup;
double-and-add is correct but ~256 doublings/scalar.

## Stage C: HMAC-SHA512 + full BIP32 child-derive kernel
**Goal**: One kernel: index → child pubkey(33) + hash160(20) + status byte.
**Tasks**:
- Host precompute: decompress parent pubkey → affine `(X,Y)`; SHA-512 states of
  `i_key_pad`/`o_key_pad` from `chain_code`. recovery already implements exactly
  this precompute (`hmac_sha512_const_precompute` / `hmac_sha512_precomp_t` in
  `DerivationFunc.cuh`) — adapt it. Pack into a kernel params struct.
- Device `derive_child(i)`: build msg `pubkey||index_be` → HMAC-SHA512 (using
  precomputed pads, from recovery) → split `IL/IR` → validity check
  (`0 < IL < n`) → `IL·G` (Stage B) → `+ parent_point` → compress →
  `hash160` (Stage A) → write outputs + status.
- NOTE: this public-side composition (`parent_pub + IL·G`, a point add) is the
  one piece NO surveyed repo implements end-to-end — recovery/brainsecp both
  derive the **private** side (`priv + IL mod n`, a scalar add). The primitives
  exist; this glue is the genuinely new code.

**Tests / Success Criteria**:
- For the repo's canonical xpub (`seed=[0u8;64]`, `m/84'/0'/0'`), GPU outputs
  for indices `0..512` match `ExtendedPublicKey::derive_child(i).key` AND
  `hash160` **byte-for-byte**.  ✅

**Status**: COMPLETE. `kernels/derive.cu` (`derive_child_kernel`) composes
HMAC-SHA512 (host-precompute deferred — full HMAC on device for now) → IL/IR →
`0 < IL < n` check → `IL·G` → `+ parent_pub` (parent decompressed on host, passed
as affine X,Y) → compress → hash160 → per-slot status byte. Device code
refactored into shared headers `secp256k1.cuh` + `hash.cuh` (now also SHA-512 +
HMAC-SHA512). Host: `DeriveKernel::derive_batch` in `src/gpu_kernel.rs`.
IR currently unused (depth-1 target). HMAC pad precompute (recovery's
optimization) folded into Stage E.

## Stage D: Host integration, streaming, and the correctness harness
**Goal**: `CudaContext::batch_derive` (the real `cuda` feature) actually
launches the kernel; remove the CPU fallback in `cuda.rs:212-234`.
**Tasks**:
- Replace the rayon block with: alloc device output buffers (already present),
  build params, `launch` with `grid = ceil(count / block_size)`, copy results
  back, repack into `Vec<[u8;33]>` / `Vec<[u8;20]>` (repack code already
  exists at `cuda.rs:268-276`).
- Chunk large `count` across `num_streams` for overlap of copy/compute.
- A `cargo test --features cuda` harness (gated to run only when a GPU is
  present) diffing GPU vs `batch_derive_cpu` over e.g. 100k indices, including a
  range chosen to exercise a forced-invalid slot if feasible.
- **Honesty fix**: relabel `cuda-sim` so it does not report fabricated GPU
  throughput as if measured (`cuda.rs:384-399`); make clear it is a CPU stand-in
  or have it run the same kernel via a CPU emulation, not invented numbers.

**Tests / Success Criteria**:
- `--features cuda` end-to-end run matches CPU, 0 mismatches.  ✅ (512-index
  test `derive_batch_matches_cpu`; `gpu-demo` runs on real hardware)
- No per-index host compute remains on the `cuda` path.  ✅ (rayon block deleted;
  `cuda.rs::batch_derive` now just launches `DeriveKernel`)

**Status**: COMPLETE. `cuda.rs::batch_derive` delegates to
`DeriveKernel::derive_batch` (real on-device compute); the CPU fallback +
pointless host↔device copies are gone. `cuda-sim` no longer fabricates GPU
throughput — it reports real measured CPU wall time with a comment that it is
NOT a GPU number. Measured on RTX 3070 Laptop: ~302k addr/s GPU vs ~124k addr/s
CPU at count=5000 (naive double-and-add; Stage E should multiply this). Streams
/ chunking for very large `count` deferred to Stage E.

## Stage E (optional): Performance
**Goal**: Approach the 10M+ addr/s target in the original plan.
**Done**: Fixed-base windowed table for `IL·G` — 8-bit windows, 32 windows ×
255 affine entries (~522 KB VRAM), built once on host via k256
(`build_gtable`), uploaded per kernel. Replaces 256 doublings + ~128 adds with
≤32 mixed adds and zero doublings. `scalar_mul_g_table` in `secp256k1.cuh`;
both `scalar_mul_g_kernel` and `derive_child_kernel` use it. Re-validated vs
k256 and CPU. Block size swept: 64 best on sm_86 (128 lowered occupancy due to
the large per-thread HMAC/EC local-memory footprint).

**Measured (RTX 3070 Laptop, release):**
| count | GPU addr/s | vs CPU (~130k) |
|-------|-----------|----------------|
| naive double-and-add | ~302k | 2.4× |
| table, 5k | ~1.1M | ~9× |
| table, 50k–200k | ~2.0–2.7M | ~16–20× |

**Remaining (open-ended, not done):** HMAC pad host-precompute (recovery's
trick — skip 2 of 4 SHA-512 compressions), batched modular inversion
(Montgomery, amortize the per-point Fermat `inv` which now dominates),
multi-stream chunking for very large counts, reducing per-thread local memory
(the real occupancy limiter). These would push toward the 10M+ target.

**Status**: Core optimization COMPLETE (table). Further tuning optional.

---

## Reuse map (repo survey, June 2026)

Every GPU primitive needed already exists in permissively-licensed repos. The
only thing none of them do is the **public-side** non-hardened derivation from
an xpub — that glue is ours. Mixing MIT + Apache-2.0 is fine; preserve both
LICENSE/NOTICE files for whatever is vendored. Do NOT copy code we don't use
(e.g. recovery's vendored ed25519-donna / fastpbkdf2 → extra licenses).

| Primitive | Source repo | File / symbol | License |
|-----------|-------------|---------------|---------|
| Fixed-base `IL·G` (windowed table accumulate) | XopMC/CudaBrainSecp | `GPU/GPUSecp.cu` `_PointMultiSecp256k1` | MIT |
| Point add (Jacobian + affine) | CudaBrainSecp | `GPU/GPUMath.h:977` `_PointAddSecp256k1` | MIT |
| Field ops `_ModAdd/Sub/Mult/Inv` | CudaBrainSecp | `GPU/GPUMath.h` | MIT |
| Compressed `hash160((x,y&1))` | CudaBrainSecp | `GPU/GPUHash.h:858` `_GetHash160Comp` | MIT |
| SHA-256 / RIPEMD-160 device fns | CudaBrainSecp | `GPU/GPUHash.h` | MIT |
| Generator table builder (host) | CudaBrainSecp | `CPU/Point.cpp`, `CPU/SECP256K1.cpp` | MIT |
| HMAC-SHA512 + host pad precompute | XopMC/CUDA_Mnemonic_Recovery | `include/cuda/DerivationFunc.cuh` `hmac_sha512_const_precompute`, `hmac_sha512_precomp_t` | Apache-2.0 |
| Alt. secp256k1 suite + `G,2G,3G…` table | CUDA_Mnemonic_Recovery | `third_party/secp256k1/GPUGroup.cuh` | Apache-2.0 |

Notes:
- recovery's top-level derive (`normal_private_child_from_private`) is **private
  side** (scalar add) — not directly reusable; its HMAC + field/group/table
  pieces are.
- brainsecp limbs are 4×u64 little-endian; bridge to the big-endian wire format
  the CPU oracle uses.
- brainsecp's 16-bit-window gTable is ~128 MB VRAM (space/time tradeoff); tune
  in Stage E.
- Neither repo uses cudarc — Rust host glue + PTX build is ours regardless.

Clone refs surveyed: `8891689/secp256k1` is a **CPU C** lib (not CUDA) — not
useful here despite search results implying otherwise.

## Verification oracle (used by every stage)

The CPU path is ground truth. Any GPU output is correct iff it equals, for the
same index, the existing pure-Rust implementation:
- `ExtendedPublicKey::derive_child(i).key` — `bip32.rs:228`
- `address::hash160(..)` — `address.rs:9`

## Risks / unknowns

1. **secp256k1 GPU arithmetic correctness** is the dominant risk — mitigated by
   Stage B's fuzzing against `k256` before any BIP32 wiring.
2. ~~No GPU available~~ — RESOLVED: this box has an RTX 3070 Laptop (sm_86) +
   nvcc 12.8 (nix). Stage A acceptance tests run and pass on real hardware.
   `cuda-sim` still must NOT be treated as validation.
3. ~~`nvcc` availability / arch flags~~ — RESOLVED in Stage A: nvcc on PATH,
   `build.rs` targets `sm_86` (override via `CUDA_ARCH`).
4. Effort: Stage B is the bulk (a small secp256k1-on-CUDA core). A/C/D are
   moderate; E is open-ended.

## Done / not in scope

- [x] BIP39, BIP32 (CPU), address encoding, CPU batch baseline, CLI — complete.
- Base58/Bech32 string encoding stays on the **host** (cheap, not worth GPU);
  the GPU returns raw `hash160`, the host formats addresses as today.
