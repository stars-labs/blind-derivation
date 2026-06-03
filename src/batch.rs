//! Batch Address Derivation
//!
//! This module provides high-performance batch derivation using CPU parallelism.
//! It serves as both a usable implementation and a baseline for CUDA comparison.

use crate::address::{generate_address, AddressType};
use crate::bip32::ExtendedPublicKey;
use rayon::prelude::*;

pub const MAX_BATCH_COUNT: u32 = 1_000_000;
pub const MAX_START_INDEX: u32 = u32::MAX - MAX_BATCH_COUNT;

/// Result of batch derivation
#[derive(Debug, Clone)]
pub struct DerivedAddress {
    pub index: u32,
    pub address: String,
    pub pubkey: [u8; 33],
}

/// Batch derivation configuration
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Starting index for derivation
    pub start_index: u32,
    /// Number of addresses to derive
    pub count: u32,
    /// Address type to generate
    pub address_type: AddressType,
    /// Use mainnet (true) or testnet (false)
    pub mainnet: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            start_index: 0,
            count: 1000,
            address_type: AddressType::P2WPKH,
            mainnet: true,
        }
    }
}

impl BatchConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.count > MAX_BATCH_COUNT {
            return Err(format!(
                "count {} exceeds maximum batch size of {}",
                self.count, MAX_BATCH_COUNT
            ));
        }
        if self.start_index > MAX_START_INDEX {
            return Err(format!(
                "start_index {} would cause index overflow",
                self.start_index
            ));
        }
        Ok(())
    }
}

/// Batch derive addresses using CPU parallelism (baseline for CUDA comparison)
///
/// # Arguments
/// * `xpub` - Extended public key to derive from
/// * `config` - Batch derivation configuration
///
/// # Returns
/// Vector of derived addresses
pub fn batch_derive_cpu(
    xpub: &ExtendedPublicKey,
    config: &BatchConfig,
) -> Vec<Result<DerivedAddress, String>> {
    if let Err(e) = config.validate() {
        return vec![Err(e)];
    }
    (config.start_index..config.start_index + config.count)
        .into_par_iter()
        .map(|index| {
            xpub.derive_child(index)
                .map(|child| DerivedAddress {
                    index,
                    address: generate_address(&child.key, config.address_type, config.mainnet),
                    pubkey: child.key,
                })
                .map_err(|e| format!("Failed to derive index {}: {}", index, e))
        })
        .collect()
}

/// Batch derive addresses sequentially (for comparison/debugging)
pub fn batch_derive_sequential(
    xpub: &ExtendedPublicKey,
    config: &BatchConfig,
) -> Vec<Result<DerivedAddress, String>> {
    if let Err(e) = config.validate() {
        return vec![Err(e)];
    }
    (config.start_index..config.start_index + config.count)
        .map(|index| {
            xpub.derive_child(index)
                .map(|child| DerivedAddress {
                    index,
                    address: generate_address(&child.key, config.address_type, config.mainnet),
                    pubkey: child.key,
                })
                .map_err(|e| format!("Failed to derive index {}: {}", index, e))
        })
        .collect()
}

/// Export xpub data for GPU processing
#[derive(Debug, Clone)]
pub struct GpuExportData {
    /// Compressed public key (33 bytes)
    pub pubkey: [u8; 33],
    /// Chain code (32 bytes)
    pub chain_code: [u8; 32],
}

impl From<&ExtendedPublicKey> for GpuExportData {
    fn from(xpub: &ExtendedPublicKey) -> Self {
        Self {
            pubkey: xpub.key,
            chain_code: xpub.chain_code,
        }
    }
}

impl GpuExportData {
    pub fn validate(&self) -> Result<(), String> {
        if self.pubkey[0] != 0x02 && self.pubkey[0] != 0x03 {
            return Err(format!(
                "invalid compressed pubkey prefix 0x{:02x}: expected 0x02 or 0x03",
                self.pubkey[0]
            ));
        }
        if self.pubkey[1..] == [0u8; 32] {
            return Err("invalid compressed pubkey: x coordinate is all-zero".to_string());
        }
        Ok(())
    }

    /// Serialize to bytes for GPU transfer
    pub fn to_bytes(&self) -> [u8; 65] {
        let mut bytes = [0u8; 65];
        bytes[..33].copy_from_slice(&self.pubkey);
        bytes[33..65].copy_from_slice(&self.chain_code);
        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8; 65]) -> Self {
        let mut pubkey = [0u8; 33];
        let mut chain_code = [0u8; 32];
        pubkey.copy_from_slice(&bytes[..33]);
        chain_code.copy_from_slice(&bytes[33..65]);
        Self { pubkey, chain_code }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bip32::ExtendedPrivateKey;

    fn test_xpub() -> ExtendedPublicKey {
        let seed = [0u8; 64];
        let master = ExtendedPrivateKey::from_seed(&seed).unwrap();
        let account = master.derive_path("m/84'/0'/0'").unwrap();
        account.public_key()
    }

    #[test]
    fn test_batch_derive_cpu() {
        let xpub = test_xpub();
        let config = BatchConfig {
            start_index: 0,
            count: 100,
            address_type: AddressType::P2WPKH,
            mainnet: true,
        };

        let results = batch_derive_cpu(&xpub, &config);
        assert_eq!(results.len(), 100);

        // All should succeed
        for result in &results {
            assert!(result.is_ok());
        }

        // First address should be deterministic
        let first = results[0].as_ref().unwrap();
        assert_eq!(first.index, 0);
        assert!(first.address.starts_with("bc1q"));
    }

    #[test]
    fn test_parallel_matches_sequential() {
        let xpub = test_xpub();
        let config = BatchConfig {
            start_index: 0,
            count: 50,
            ..Default::default()
        };

        let parallel = batch_derive_cpu(&xpub, &config);
        let sequential = batch_derive_sequential(&xpub, &config);

        for (p, s) in parallel.iter().zip(sequential.iter()) {
            let p = p.as_ref().unwrap();
            let s = s.as_ref().unwrap();
            assert_eq!(p.index, s.index);
            assert_eq!(p.address, s.address);
            assert_eq!(p.pubkey, s.pubkey);
        }
    }

    #[test]
    fn test_gpu_export_roundtrip() {
        let xpub = test_xpub();
        let export: GpuExportData = (&xpub).into();
        let bytes = export.to_bytes();
        let restored = GpuExportData::from_bytes(&bytes);

        assert_eq!(export.pubkey, restored.pubkey);
        assert_eq!(export.chain_code, restored.chain_code);
    }

    #[test]
    fn test_batch_config_validate_ok() {
        let config = BatchConfig {
            start_index: 0,
            count: MAX_BATCH_COUNT,
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_batch_config_validate_count_too_large() {
        let config = BatchConfig {
            start_index: 0,
            count: MAX_BATCH_COUNT + 1,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.contains("exceeds maximum"));
    }

    #[test]
    fn test_batch_config_validate_start_index_overflow() {
        let config = BatchConfig {
            start_index: MAX_START_INDEX + 1,
            count: 1,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.contains("overflow"));
    }

    #[test]
    fn test_batch_derive_cpu_rejects_oversized_batch() {
        let xpub = test_xpub();
        let config = BatchConfig {
            start_index: 0,
            count: MAX_BATCH_COUNT + 1,
            ..Default::default()
        };
        let results = batch_derive_cpu(&xpub, &config);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_err());
    }

    #[test]
    fn test_batch_derive_sequential_rejects_oversized_batch() {
        let xpub = test_xpub();
        let config = BatchConfig {
            start_index: 0,
            count: MAX_BATCH_COUNT + 1,
            ..Default::default()
        };
        let results = batch_derive_sequential(&xpub, &config);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_err());
    }

    #[test]
    fn test_gpu_export_data_validate_ok() {
        let xpub = test_xpub();
        let export: GpuExportData = (&xpub).into();
        assert!(export.validate().is_ok());
    }

    #[test]
    fn test_gpu_export_data_validate_bad_prefix() {
        let mut export = GpuExportData {
            pubkey: [0u8; 33],
            chain_code: [0u8; 32],
        };
        export.pubkey[0] = 0x04;
        export.pubkey[1] = 0x01; // non-zero x coord
        let err = export.validate().unwrap_err();
        assert!(err.contains("0x04"));
    }

    #[test]
    fn test_gpu_export_data_validate_zero_x_coord() {
        // Build one with valid prefix but all-zero x coordinate
        let mut export = GpuExportData {
            pubkey: [0u8; 33],
            chain_code: [0u8; 32],
        };
        export.pubkey[0] = 0x02;
        // pubkey[1..] remains all-zero
        let err = export.validate().unwrap_err();
        assert!(err.contains("all-zero"));
    }
}
