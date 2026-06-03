//! Blind Derivation - HD Wallet Batch Address Derivation
//!
//! This library provides secure, high-performance HD wallet address derivation
//! with support for GPU acceleration.
//!
//! ## Architecture
//!
//! The derivation is split into two phases:
//!
//! 1. **Local (Secure)**: Mnemonic → Seed → Extended Private Key → Extended Public Key
//!    - Never leaves the local machine
//!    - Uses hardened derivation paths
//!
//! 2. **Remote (GPU-acceleratable)**: Extended Public Key → Batch Addresses
//!    - Can be outsourced to GPU clusters
//!    - Uses non-hardened derivation only
//!    - Cannot reverse to get private keys
//!
//! ## Example
//!
//! ```rust,ignore
//! use blind_derivation::{bip39, bip32, batch, address::AddressType};
//!
//! // Step 1: Local - derive xpub from mnemonic
//! let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
//! let seed = bip39::mnemonic_to_seed(mnemonic, "").unwrap();
//!
//! let master = bip32::ExtendedPrivateKey::from_seed(&seed).unwrap();
//! let account = master.derive_path("m/84'/0'/0'").unwrap();
//! let xpub = account.public_key();
//!
//! // Step 2: Export xpub (safe to share for non-hardened derivation)
//! println!("xpub: {}", xpub.to_base58());
//!
//! // Step 3: Batch derive addresses (can be done on GPU)
//! let config = batch::BatchConfig {
//!     start_index: 0,
//!     count: 1000,
//!     address_type: AddressType::P2WPKH,
//!     mainnet: true,
//! };
//!
//! let addresses = batch::batch_derive_cpu(&xpub, &config);
//! ```

pub mod address;
pub mod batch;
pub mod bip32;
pub mod bip39;
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod gpu_kernel;

// Re-exports for convenience
pub use address::AddressType;
pub use batch::{batch_derive_cpu, BatchConfig, DerivedAddress, GpuExportData};
pub use bip32::{ExtendedPrivateKey, ExtendedPublicKey};
pub use bip39::mnemonic_to_seed;
pub use cuda::{CudaBatchConfig, CudaBatchResult, CudaContext, CudaDeviceInfo, CudaError};
