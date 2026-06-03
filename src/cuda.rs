//! CUDA Batch Address Derivation
//!
//! This module provides GPU-accelerated batch address derivation using CUDA.
//!
//! # Features
//!
//! - `cuda` - Real CUDA support via cudarc (requires CUDA toolkit)
//! - `cuda-sim` - Simulation mode for demo without GPU hardware
//!
//! # Building
//!
//! ```bash
//! # With real CUDA (requires CUDA toolkit)
//! cargo build --release --features cuda
//!
//! # Simulation mode (no GPU required)
//! cargo build --release --features cuda-sim
//! ```

use crate::batch::GpuExportData;
use thiserror::Error;

/// CUDA device information
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    pub name: String,
    pub compute_capability: (u32, u32),
    pub total_memory: u64,
    pub multiprocessor_count: u32,
}

/// CUDA batch derivation configuration
#[derive(Debug, Clone)]
pub struct CudaBatchConfig {
    pub start_index: u32,
    pub count: u32,
    pub block_size: u32,
    pub num_streams: u32,
}

impl Default for CudaBatchConfig {
    fn default() -> Self {
        Self {
            start_index: 0,
            count: 1_000_000,
            block_size: 256,
            num_streams: 4,
        }
    }
}

/// Result from CUDA batch derivation
#[derive(Debug)]
pub struct CudaBatchResult {
    pub count: u32,
    pub pubkeys: Vec<[u8; 33]>,
    pub address_hashes: Vec<[u8; 20]>,
    pub elapsed_ms: f64,
}

/// Error types for CUDA operations
#[derive(Debug, Error)]
pub enum CudaError {
    #[error("CUDA not available: {0}")]
    NotAvailable(String),
    #[error("CUDA driver error: {0}")]
    DriverError(String),
    #[error("Kernel error: {0}")]
    KernelError(String),
    #[error("Memory error: {0}")]
    MemoryError(String),
}

// ============================================================================
// Real CUDA Implementation (when cuda feature is enabled)
// ============================================================================

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use crate::gpu_kernel::DeriveKernel;
    use cudarc::driver::sys::{
        cuDeviceGet, cuDeviceGetAttribute, cuDeviceGetName, cuDeviceTotalMem_v2,
        CUdevice_attribute,
    };
    use cudarc::driver::{CudaContext as CudarContext, CudaStream};
    use std::sync::Arc;
    use std::time::Instant;

    pub struct CudaContext {
        pub ctx: Arc<CudarContext>,
        pub stream: Arc<CudaStream>,
        pub device_info: CudaDeviceInfo,
        /// Real BIP32 derive kernel (loaded PTX). The actual GPU compute path.
        derive: DeriveKernel,
    }

    fn get_device_info(device_id: i32) -> Result<CudaDeviceInfo, CudaError> {
        unsafe {
            let mut device: i32 = 0;
            let result = cuDeviceGet(&mut device, device_id);
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(CudaError::DriverError(format!(
                    "cuDeviceGet failed: {:?}",
                    result
                )));
            }

            let mut name_buf = [0i8; 256];
            let result = cuDeviceGetName(name_buf.as_mut_ptr(), 256, device);
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(CudaError::DriverError(format!(
                    "cuDeviceGetName failed: {:?}",
                    result
                )));
            }
            let name = std::ffi::CStr::from_ptr(name_buf.as_ptr())
                .to_string_lossy()
                .into_owned();

            let mut total_mem: usize = 0;
            let result = cuDeviceTotalMem_v2(&mut total_mem, device);
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(CudaError::DriverError(format!(
                    "cuDeviceTotalMem failed: {:?}",
                    result
                )));
            }

            let mut major: i32 = 0;
            let mut minor: i32 = 0;
            cuDeviceGetAttribute(
                &mut major,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                device,
            );
            cuDeviceGetAttribute(
                &mut minor,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                device,
            );

            let mut mp_count: i32 = 0;
            cuDeviceGetAttribute(
                &mut mp_count,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                device,
            );

            Ok(CudaDeviceInfo {
                name,
                compute_capability: (major as u32, minor as u32),
                total_memory: total_mem as u64,
                multiprocessor_count: mp_count as u32,
            })
        }
    }

    impl CudaContext {
        pub fn new(device_id: usize) -> Result<Self, CudaError> {
            let ctx = CudarContext::new(device_id).map_err(|e| {
                CudaError::DriverError(format!(
                    "Failed to create context for device {}: {}",
                    device_id, e
                ))
            })?;

            let device_info = get_device_info(device_id as i32)?;
            let stream = ctx.default_stream();
            let derive = DeriveKernel::new(device_id)?;

            Ok(Self {
                ctx,
                stream,
                device_info,
                derive,
            })
        }

        pub fn list_devices() -> Result<Vec<CudaDeviceInfo>, CudaError> {
            let mut devices = Vec::new();
            for i in 0..16 {
                match CudaContext::new(i) {
                    Ok(ctx) => devices.push(ctx.device_info),
                    Err(_) => break,
                }
            }
            if devices.is_empty() {
                return Err(CudaError::NotAvailable("No CUDA devices found".to_string()));
            }
            Ok(devices)
        }

        /// Real GPU batch derivation: launches the BIP32 derive kernel and
        /// returns child pubkeys + HASH160 computed entirely on-device.
        pub fn batch_derive(
            &self,
            xpub: &GpuExportData,
            config: &CudaBatchConfig,
        ) -> Result<CudaBatchResult, CudaError> {
            xpub.validate().map_err(CudaError::KernelError)?;
            let start = Instant::now();

            let derived = self.derive.derive_batch(
                &xpub.pubkey,
                &xpub.chain_code,
                config.start_index,
                config.count,
            )?;

            let pubkeys: Vec<[u8; 33]> = derived.iter().map(|d| d.pubkey).collect();
            let address_hashes: Vec<[u8; 20]> = derived.iter().map(|d| d.hash160).collect();
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

            Ok(CudaBatchResult {
                count: config.count,
                pubkeys,
                address_hashes,
                elapsed_ms,
            })
        }

        pub fn test_gpu(&self) -> Result<String, CudaError> {
            let test_size = 1024usize;
            let _d_test = self
                .stream
                .alloc_zeros::<f32>(test_size)
                .map_err(|e| CudaError::MemoryError(format!("alloc failed: {}", e)))?;

            self.stream
                .synchronize()
                .map_err(|e| CudaError::KernelError(format!("sync failed: {}", e)))?;

            Ok(format!(
                "GPU test passed on {} (SM {}.{}, {} MB, {} MPs)",
                self.device_info.name,
                self.device_info.compute_capability.0,
                self.device_info.compute_capability.1,
                self.device_info.total_memory / 1024 / 1024,
                self.device_info.multiprocessor_count
            ))
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::CudaContext;

// ============================================================================
// Simulation Implementation (when cuda-sim feature is enabled)
// ============================================================================

#[cfg(all(feature = "cuda-sim", not(feature = "cuda")))]
mod sim_impl {
    use super::*;
    use crate::address::hash160;
    use crate::bip32::ExtendedPublicKey;
    use rayon::prelude::*;
    use std::time::Instant;

    pub struct CudaContext {
        pub device_info: CudaDeviceInfo,
    }

    impl CudaContext {
        pub fn new(device_id: usize) -> Result<Self, CudaError> {
            let device_info = CudaDeviceInfo {
                name: format!("Simulated GPU {} (RTX 4090)", device_id),
                compute_capability: (8, 9),
                total_memory: 24 * 1024 * 1024 * 1024, // 24 GB
                multiprocessor_count: 128,
            };

            Ok(Self { device_info })
        }

        pub fn list_devices() -> Result<Vec<CudaDeviceInfo>, CudaError> {
            Ok(vec![CudaDeviceInfo {
                name: "Simulated GPU 0 (RTX 4090)".to_string(),
                compute_capability: (8, 9),
                total_memory: 24 * 1024 * 1024 * 1024,
                multiprocessor_count: 128,
            }])
        }

        pub fn batch_derive(
            &self,
            xpub: &GpuExportData,
            config: &CudaBatchConfig,
        ) -> Result<CudaBatchResult, CudaError> {
            let start = Instant::now();

            // Reconstruct an ExtendedPublicKey from the exported GPU data.
            // depth/parent_fingerprint/child_number are metadata only; derivation
            // relies solely on key and chain_code.
            let ext_pub = ExtendedPublicKey {
                key: xpub.pubkey,
                chain_code: xpub.chain_code,
                depth: 0,
                parent_fingerprint: [0; 4],
                child_number: 0,
            };

            // Derive all child keys in parallel (simulating GPU parallelism).
            let range: Vec<u32> = (config.start_index..config.start_index + config.count).collect();
            let derived: Vec<([u8; 33], [u8; 20])> = range
                .into_par_iter()
                .map(|i| {
                    match ext_pub.derive_child(i) {
                        Ok(child) => (child.key, hash160(&child.key)),
                        Err(_) => ([0u8; 33], [0u8; 20]),
                    }
                })
                .collect();

            let (pubkeys, address_hashes): (Vec<[u8; 33]>, Vec<[u8; 20]>) =
                derived.into_iter().unzip();

            // `cuda-sim` computes on the CPU; report the REAL measured CPU wall
            // time, not a fabricated GPU throughput. This is a correctness
            // stand-in for environments without a GPU — it must never be quoted
            // as GPU performance. Use `--features cuda` for real timings.
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

            Ok(CudaBatchResult {
                count: config.count,
                pubkeys,
                address_hashes,
                elapsed_ms,
            })
        }

        pub fn test_gpu(&self) -> Result<String, CudaError> {
            Ok(format!(
                "[SIMULATION] GPU test passed on {} (SM {}.{}, {} MB, {} MPs)",
                self.device_info.name,
                self.device_info.compute_capability.0,
                self.device_info.compute_capability.1,
                self.device_info.total_memory / 1024 / 1024,
                self.device_info.multiprocessor_count
            ))
        }
    }
}

#[cfg(all(feature = "cuda-sim", not(feature = "cuda")))]
pub use sim_impl::CudaContext;

// ============================================================================
// Stub Implementation (when neither cuda nor cuda-sim is enabled)
// ============================================================================

#[cfg(not(any(feature = "cuda", feature = "cuda-sim")))]
pub struct CudaContext {
    pub device_info: CudaDeviceInfo,
}

#[cfg(not(any(feature = "cuda", feature = "cuda-sim")))]
impl CudaContext {
    pub fn new(_device_id: usize) -> Result<Self, CudaError> {
        Err(CudaError::NotAvailable(
            "CUDA not enabled. Build with: --features cuda (real GPU) or --features cuda-sim (simulation)".to_string(),
        ))
    }

    pub fn list_devices() -> Result<Vec<CudaDeviceInfo>, CudaError> {
        Err(CudaError::NotAvailable(
            "CUDA not enabled. Build with: --features cuda or --features cuda-sim".to_string(),
        ))
    }

    pub fn batch_derive(
        &self,
        _xpub: &GpuExportData,
        _config: &CudaBatchConfig,
    ) -> Result<CudaBatchResult, CudaError> {
        Err(CudaError::NotAvailable("CUDA not enabled".to_string()))
    }

    pub fn test_gpu(&self) -> Result<String, CudaError> {
        Err(CudaError::NotAvailable("CUDA not enabled".to_string()))
    }
}

/// Performance estimates
pub mod performance {
    pub const RTX_4090_ESTIMATE: u64 = 20_000_000;
    pub const RTX_3090_ESTIMATE: u64 = 15_000_000;
    pub const A100_ESTIMATE: u64 = 25_000_000;

    pub fn estimate_time_seconds(count: u64, gpu_rate: u64) -> f64 {
        count as f64 / gpu_rate as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = CudaBatchConfig::default();
        assert_eq!(config.count, 1_000_000);
        assert_eq!(config.block_size, 256);
    }

    #[test]
    fn test_performance_estimate() {
        let time = performance::estimate_time_seconds(100_000_000, performance::RTX_4090_ESTIMATE);
        assert!(time < 10.0);
    }
}
