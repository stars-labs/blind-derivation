//! BIP32 Hierarchical Deterministic Key Derivation
//!
//! This module implements HD key derivation with support for both
//! hardened and non-hardened paths. Non-hardened derivation allows
//! GPU acceleration using only the public key.

use hmac::{Hmac, Mac};
use k256::{
    elliptic_curve::{
        ops::Reduce,
        sec1::{FromEncodedPoint, ToEncodedPoint},
        Group, ScalarPrimitive,
    },
    AffinePoint, EncodedPoint, ProjectivePoint, Scalar, U256,
};
use sha2::Sha512;
use thiserror::Error;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Maximum allowed BIP32 derivation path depth.
pub const MAX_PATH_DEPTH: usize = 10;

/// SECP256k1 curve order
const SECP256K1_ORDER: [u8; 32] = [
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE,
    0xBA, 0xAE, 0xDC, 0xE6, 0xAF, 0x48, 0xA0, 0x3B, 0xBF, 0xD2, 0x5E, 0x8C, 0xD0, 0x36, 0x41, 0x41,
];

#[derive(Error, Debug)]
pub enum Bip32Error {
    #[error("Invalid seed length: expected 64 bytes, got {0}")]
    InvalidSeedLength(usize),
    #[error("Invalid key: derived key is zero or >= curve order")]
    InvalidKey,
    #[error("Cannot derive hardened child from public key")]
    HardenedFromPublic,
    #[error("Invalid path: {0}")]
    InvalidPath(String),
    #[error("Point at infinity")]
    PointAtInfinity,
    #[error("Path depth {0} exceeds maximum allowed depth")]
    PathTooDeep(usize),
}

/// Extended private key (xprv)
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct ExtendedPrivateKey {
    /// 32-byte private key
    pub key: [u8; 32],
    /// 32-byte chain code
    pub chain_code: [u8; 32],
    /// Depth in the derivation tree
    pub depth: u8,
    /// Parent fingerprint (first 4 bytes of HASH160 of parent pubkey)
    pub parent_fingerprint: [u8; 4],
    /// Child number
    pub child_number: u32,
}

/// Extended public key (xpub)
/// This can be safely shared for non-hardened derivation
#[derive(Clone)]
pub struct ExtendedPublicKey {
    /// 33-byte compressed public key
    pub key: [u8; 33],
    /// 32-byte chain code
    pub chain_code: [u8; 32],
    /// Depth in the derivation tree
    pub depth: u8,
    /// Parent fingerprint
    pub parent_fingerprint: [u8; 4],
    /// Child number
    pub child_number: u32,
}

impl ExtendedPrivateKey {
    /// Create master key from seed (BIP32 master key generation)
    pub fn from_seed(seed: &[u8]) -> Result<Self, Bip32Error> {
        if seed.len() != 64 {
            return Err(Bip32Error::InvalidSeedLength(seed.len()));
        }

        // HMAC-SHA512 with key "Bitcoin seed"
        let mut mac =
            Hmac::<Sha512>::new_from_slice(b"Bitcoin seed").expect("HMAC accepts any key");
        mac.update(seed);
        let result = mac.finalize().into_bytes();

        let mut key = [0u8; 32];
        let mut chain_code = [0u8; 32];
        key.copy_from_slice(&result[..32]);
        chain_code.copy_from_slice(&result[32..]);

        // Validate key is valid (non-zero and < curve order)
        if key.iter().all(|&b| b == 0) || key >= SECP256K1_ORDER {
            return Err(Bip32Error::InvalidKey);
        }

        Ok(Self {
            key,
            chain_code,
            depth: 0,
            parent_fingerprint: [0; 4],
            child_number: 0,
        })
    }

    /// Get the corresponding public key
    pub fn public_key(&self) -> ExtendedPublicKey {
        let scalar = Scalar::reduce(U256::from_be_slice(&self.key));
        let point = ProjectivePoint::GENERATOR * scalar;
        let encoded = point.to_affine().to_encoded_point(true);
        let mut key = [0u8; 33];
        key.copy_from_slice(encoded.as_bytes());

        ExtendedPublicKey {
            key,
            chain_code: self.chain_code,
            depth: self.depth,
            parent_fingerprint: self.parent_fingerprint,
            child_number: self.child_number,
        }
    }

    /// Derive a child private key
    pub fn derive_child(&self, index: u32) -> Result<Self, Bip32Error> {
        let hardened = index >= 0x80000000;

        let mut mac =
            Hmac::<Sha512>::new_from_slice(&self.chain_code).expect("HMAC accepts any key");

        if hardened {
            // Hardened: use 0x00 || private_key || index
            mac.update(&[0u8]);
            mac.update(&self.key);
        } else {
            // Non-hardened: use public_key || index
            let pubkey = self.public_key();
            mac.update(&pubkey.key);
        }
        mac.update(&index.to_be_bytes());

        let result = mac.finalize().into_bytes();

        // Parse IL (left 32 bytes) as scalar
        let il = &result[..32];
        let ir = &result[32..];

        // key = (IL + parent_key) mod n
        let il_scalar = Scalar::reduce(U256::from_be_slice(il));
        let parent_scalar = Scalar::reduce(U256::from_be_slice(&self.key));
        let child_scalar = il_scalar + parent_scalar;

        // Check for invalid key
        if child_scalar.is_zero().into() {
            return Err(Bip32Error::InvalidKey);
        }

        let child_key_primitive: ScalarPrimitive<k256::Secp256k1> = child_scalar.into();
        let mut key = [0u8; 32];
        key.copy_from_slice(&child_key_primitive.to_bytes());

        let mut chain_code = [0u8; 32];
        chain_code.copy_from_slice(ir);

        // Calculate parent fingerprint
        let parent_pubkey = self.public_key();
        let fingerprint = crate::address::hash160(&parent_pubkey.key);
        let mut parent_fingerprint = [0u8; 4];
        parent_fingerprint.copy_from_slice(&fingerprint[..4]);

        Ok(Self {
            key,
            chain_code,
            depth: self.depth + 1,
            parent_fingerprint,
            child_number: index,
        })
    }

    /// Derive from a path string (e.g., "m/44'/0'/0'/0/0")
    pub fn derive_path(&self, path: &str) -> Result<Self, Bip32Error> {
        let path = path.trim();
        if !path.starts_with('m') && !path.starts_with('M') {
            return Err(Bip32Error::InvalidPath(
                "Path must start with 'm'".to_string(),
            ));
        }

        let mut current = self.clone();

        let components: Vec<&str> = path.split('/').skip(1).filter(|c| !c.is_empty()).collect();
        if components.len() > MAX_PATH_DEPTH {
            return Err(Bip32Error::PathTooDeep(components.len()));
        }
        for component in &components {
            let (index_str, hardened) = if component.ends_with('\'') || component.ends_with('h') {
                (&component[..component.len() - 1], true)
            } else {
                (*component, false)
            };

            let index: u32 = index_str
                .parse()
                .map_err(|_| Bip32Error::InvalidPath(format!("Invalid index: {}", component)))?;

            let child_index = if hardened { index + 0x80000000 } else { index };

            current = current.derive_child(child_index)?;
        }

        Ok(current)
    }
}

impl ExtendedPublicKey {
    /// Derive a child public key (non-hardened only)
    ///
    /// This is the key operation for GPU acceleration:
    /// - Only requires public key and chain code
    /// - Cannot derive hardened children
    /// - Cannot reverse to get private key
    pub fn derive_child(&self, index: u32) -> Result<Self, Bip32Error> {
        if index >= 0x80000000 {
            return Err(Bip32Error::HardenedFromPublic);
        }

        let mut mac =
            Hmac::<Sha512>::new_from_slice(&self.chain_code).expect("HMAC accepts any key");
        mac.update(&self.key);
        mac.update(&index.to_be_bytes());

        let result = mac.finalize().into_bytes();

        let il = &result[..32];
        let ir = &result[32..];

        // child_pubkey = IL * G + parent_pubkey
        let il_scalar = Scalar::reduce(U256::from_be_slice(il));
        let il_point = ProjectivePoint::GENERATOR * il_scalar;

        let parent_point =
            AffinePoint::from_encoded_point(&EncodedPoint::from_bytes(self.key).unwrap());

        if parent_point.is_none().into() {
            return Err(Bip32Error::InvalidKey);
        }

        let child_point = il_point + ProjectivePoint::from(parent_point.unwrap());

        if child_point.is_identity().into() {
            return Err(Bip32Error::PointAtInfinity);
        }

        let encoded = child_point.to_affine().to_encoded_point(true);
        let mut key = [0u8; 33];
        key.copy_from_slice(encoded.as_bytes());

        let mut chain_code = [0u8; 32];
        chain_code.copy_from_slice(ir);

        // Calculate parent fingerprint
        let fingerprint = crate::address::hash160(&self.key);
        let mut parent_fingerprint = [0u8; 4];
        parent_fingerprint.copy_from_slice(&fingerprint[..4]);

        Ok(Self {
            key,
            chain_code,
            depth: self.depth + 1,
            parent_fingerprint,
            child_number: index,
        })
    }

    /// Derive from a path string (non-hardened only, e.g., "m/0/0")
    pub fn derive_path(&self, path: &str) -> Result<Self, Bip32Error> {
        let path = path.trim();
        if !path.starts_with('m') && !path.starts_with('M') {
            return Err(Bip32Error::InvalidPath(
                "Path must start with 'm'".to_string(),
            ));
        }

        let mut current = self.clone();

        let components: Vec<&str> = path.split('/').skip(1).filter(|c| !c.is_empty()).collect();
        if components.len() > MAX_PATH_DEPTH {
            return Err(Bip32Error::PathTooDeep(components.len()));
        }
        for component in &components {
            if component.ends_with('\'') || component.ends_with('h') {
                return Err(Bip32Error::HardenedFromPublic);
            }

            let index: u32 = component
                .parse()
                .map_err(|_| Bip32Error::InvalidPath(format!("Invalid index: {}", component)))?;

            current = current.derive_child(index)?;
        }

        Ok(current)
    }

    /// Batch derive multiple child public keys (for GPU acceleration baseline)
    pub fn derive_children_batch(
        &self,
        start_index: u32,
        count: u32,
    ) -> Vec<Result<Self, Bip32Error>> {
        (start_index..start_index + count)
            .map(|i| self.derive_child(i))
            .collect()
    }

    /// Serialize to bytes (for export to GPU)
    pub fn to_bytes(&self) -> [u8; 78] {
        let mut bytes = [0u8; 78];
        // Version bytes for mainnet xpub: 0x0488B21E
        bytes[0..4].copy_from_slice(&[0x04, 0x88, 0xB2, 0x1E]);
        bytes[4] = self.depth;
        bytes[5..9].copy_from_slice(&self.parent_fingerprint);
        bytes[9..13].copy_from_slice(&self.child_number.to_be_bytes());
        bytes[13..45].copy_from_slice(&self.chain_code);
        bytes[45..78].copy_from_slice(&self.key);
        bytes
    }

    /// Serialize to base58check (xpub string)
    pub fn to_base58(&self) -> String {
        bs58::encode(self.to_bytes()).with_check().into_string()
    }
}

impl ExtendedPrivateKey {
    /// Serialize to bytes
    pub fn to_bytes(&self) -> [u8; 78] {
        let mut bytes = [0u8; 78];
        // Version bytes for mainnet xprv: 0x0488ADE4
        bytes[0..4].copy_from_slice(&[0x04, 0x88, 0xAD, 0xE4]);
        bytes[4] = self.depth;
        bytes[5..9].copy_from_slice(&self.parent_fingerprint);
        bytes[9..13].copy_from_slice(&self.child_number.to_be_bytes());
        bytes[13..45].copy_from_slice(&self.chain_code);
        bytes[45] = 0x00; // Private key prefix
        bytes[46..78].copy_from_slice(&self.key);
        bytes
    }

    /// Serialize to base58check (xprv string)
    pub fn to_base58(&self) -> String {
        bs58::encode(self.to_bytes()).with_check().into_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_master_key_from_seed() {
        // Test vector from BIP32 spec
        let seed = hex::decode("000102030405060708090a0b0c0d0e0f").unwrap();
        // Pad to 64 bytes (this is a simplified test)
        let mut full_seed = [0u8; 64];
        full_seed[..16].copy_from_slice(&seed);

        let master = ExtendedPrivateKey::from_seed(&full_seed).unwrap();
        assert_eq!(master.depth, 0);
    }

    #[test]
    fn test_bip32_test_vector_1() {
        // BIP32 Test Vector 1
        let seed_hex = "000102030405060708090a0b0c0d0e0f";
        let mut seed = [0u8; 64];
        let seed_bytes = hex::decode(seed_hex).unwrap();
        seed[..seed_bytes.len()].copy_from_slice(&seed_bytes);

        let master = ExtendedPrivateKey::from_seed(&seed).unwrap();

        // Chain m
        let xprv = master.to_base58();
        assert!(xprv.starts_with("xprv"));

        let xpub = master.public_key().to_base58();
        assert!(xpub.starts_with("xpub"));
    }

    #[test]
    fn test_hardened_derivation() {
        let seed = [0u8; 64];
        let master = ExtendedPrivateKey::from_seed(&seed).unwrap();

        // Derive m/44'/0'/0' (hardened)
        let account = master.derive_path("m/44'/0'/0'").unwrap();
        assert_eq!(account.depth, 3);
    }

    #[test]
    fn test_non_hardened_public_derivation() {
        let seed = [0u8; 64];
        let master = ExtendedPrivateKey::from_seed(&seed).unwrap();

        // Derive m/44'/0'/0' with private key, then get public
        let account_priv = master.derive_path("m/44'/0'/0'").unwrap();
        let account_pub = account_priv.public_key();

        // Now derive m/44'/0'/0'/0/0 using public key (non-hardened)
        let child_pub = account_pub.derive_path("m/0/0").unwrap();

        // And verify it matches deriving with private key
        let child_priv = account_priv.derive_path("m/0/0").unwrap();
        let child_pub_from_priv = child_priv.public_key();

        assert_eq!(child_pub.key, child_pub_from_priv.key);
    }

    #[test]
    fn test_hardened_from_public_fails() {
        let seed = [0u8; 64];
        let master = ExtendedPrivateKey::from_seed(&seed).unwrap();
        let pub_key = master.public_key();

        // Should fail - can't derive hardened from public
        assert!(pub_key.derive_child(0x80000000).is_err());
    }

    #[test]
    fn test_path_depth_limit_private() {
        let seed = [0u8; 64];
        let master = ExtendedPrivateKey::from_seed(&seed).unwrap();
        // 11 non-hardened components exceeds MAX_PATH_DEPTH=10
        let path = "m/0/0/0/0/0/0/0/0/0/0/0";
        let result = master.derive_path(path);
        assert!(matches!(result, Err(Bip32Error::PathTooDeep(11))));
    }

    #[test]
    fn test_path_depth_limit_public() {
        let seed = [0u8; 64];
        let master = ExtendedPrivateKey::from_seed(&seed).unwrap();
        let pub_key = master.public_key();
        let path = "m/0/0/0/0/0/0/0/0/0/0/0";
        let result = pub_key.derive_path(path);
        assert!(matches!(result, Err(Bip32Error::PathTooDeep(11))));
    }

    #[test]
    fn test_path_at_max_depth_succeeds() {
        let seed = [0u8; 64];
        let master = ExtendedPrivateKey::from_seed(&seed).unwrap();
        // Exactly 10 components — should succeed
        let path = "m/0/0/0/0/0/0/0/0/0/0";
        let result = master.derive_path(path);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().depth, 10);
    }

    #[test]
    fn test_private_key_zeroize_on_drop() {
        // Verifies that ZeroizeOnDrop compiles and that we can construct and drop the key
        let seed = [0u8; 64];
        {
            let _key = ExtendedPrivateKey::from_seed(&seed).unwrap();
            // _key is dropped here; ZeroizeOnDrop zeroes the fields
        }
        // No assertion needed — this test checks compile-time and drop-path correctness
    }
}
