//! BIP39 Mnemonic to Seed derivation
//!
//! This module handles the secure conversion of a 24-word mnemonic
//! to a 512-bit seed using PBKDF2-HMAC-SHA512.

use hmac::Hmac;
use pbkdf2::pbkdf2;
use sha2::{Digest, Sha256, Sha512};
use thiserror::Error;
use zeroize::Zeroizing;

/// BIP39 wordlist (English)
pub const WORDLIST: &[&str] = &include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/wordlist.txt"));

#[derive(Error, Debug)]
pub enum Bip39Error {
    #[error("Invalid mnemonic word: {0}")]
    InvalidWord(String),
    #[error("Invalid mnemonic length: expected 12, 15, 18, 21, or 24 words, got {0}")]
    InvalidLength(usize),
    #[error("Invalid checksum")]
    InvalidChecksum,
}

/// Validates a mnemonic phrase, including BIP39 checksum verification.
pub fn validate_mnemonic(mnemonic: &str) -> Result<Vec<&str>, Bip39Error> {
    let words: Vec<&str> = mnemonic.split_whitespace().collect();

    // Check word count
    let valid_lengths = [12, 15, 18, 21, 24];
    if !valid_lengths.contains(&words.len()) {
        return Err(Bip39Error::InvalidLength(words.len()));
    }

    // Validate each word and collect its 11-bit index in the wordlist
    let mut indices = Vec::with_capacity(words.len());
    for word in &words {
        match WORDLIST.iter().position(|&w| w == *word) {
            Some(idx) => indices.push(idx as u16),
            None => return Err(Bip39Error::InvalidWord(word.to_string())),
        }
    }

    // Pack word indices into a contiguous bit array.
    // Each word contributes 11 bits; total = words.len() * 11.
    // Layout: [entropy bits | checksum bits]
    //   checksum_bits = total_bits / 33
    //   entropy_bits  = total_bits - checksum_bits  (always a multiple of 8)
    let total_bits = words.len() * 11;
    let checksum_bits = total_bits / 33;
    let entropy_bytes = (total_bits - checksum_bits) / 8;

    let mut bit_data = vec![0u8; (total_bits + 7) / 8];
    for (i, &idx) in indices.iter().enumerate() {
        for j in 0..11usize {
            let bit = (idx >> (10 - j)) & 1;
            let pos = i * 11 + j;
            if bit != 0 {
                bit_data[pos / 8] |= 1 << (7 - (pos % 8));
            }
        }
    }

    // Verify checksum: first `checksum_bits` bits of SHA256(entropy)
    // must equal the checksum bits stored at the end of bit_data.
    let entropy = &bit_data[..entropy_bytes];
    let hash = Sha256::digest(entropy);

    let stored_cs = bit_data[entropy_bytes] >> (8 - checksum_bits);
    let computed_cs = hash[0] >> (8 - checksum_bits);

    if stored_cs != computed_cs {
        return Err(Bip39Error::InvalidChecksum);
    }

    Ok(words)
}

/// Derives a 512-bit seed from a mnemonic phrase
///
/// Uses PBKDF2-HMAC-SHA512 with 2048 iterations as per BIP39 spec.
/// The passphrase is optional (empty string if not provided).
pub fn mnemonic_to_seed(mnemonic: &str, passphrase: &str) -> Result<Zeroizing<[u8; 64]>, Bip39Error> {
    // Validate mnemonic first
    validate_mnemonic(mnemonic)?;

    // BIP39 salt is "mnemonic" + passphrase
    let salt = format!("mnemonic{}", passphrase);

    let mut seed = Zeroizing::new([0u8; 64]);
    pbkdf2::<Hmac<Sha512>>(mnemonic.as_bytes(), salt.as_bytes(), 2048, &mut *seed)
        .expect("PBKDF2 should not fail with valid parameters");

    Ok(seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mnemonic_to_seed_vector() {
        // Test vector from BIP39 spec
        let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
        let seed = mnemonic_to_seed(mnemonic, "").unwrap();

        // Expected seed (first 32 bytes shown for brevity)
        let expected_hex = "5eb00bbddcf069084889a8ab9155568165f5c453ccb85e70811aaed6f6da5fc19a5ac40b389cd370d086206dec8aa6c43daea6690f20ad3d8d48b2d2ce9e38e4";
        assert_eq!(hex::encode(&*seed), expected_hex);
    }

    #[test]
    fn test_mnemonic_to_seed_with_passphrase() {
        let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
        let seed = mnemonic_to_seed(mnemonic, "TREZOR").unwrap();

        let expected_hex = "c55257c360c07c72029aebc1b53c05ed0362ada38ead3e3e9efa3708e53495531f09a6987599d18264c1e1c92f2cf141630c7a3c4ab7c81b2f001698e7463b04";
        assert_eq!(hex::encode(&*seed), expected_hex);
    }

    #[test]
    fn test_invalid_word() {
        let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon invalid";
        assert!(matches!(
            mnemonic_to_seed(mnemonic, ""),
            Err(Bip39Error::InvalidWord(_))
        ));
    }

    #[test]
    fn test_invalid_length() {
        let mnemonic = "abandon abandon abandon";
        assert!(matches!(
            mnemonic_to_seed(mnemonic, ""),
            Err(Bip39Error::InvalidLength(3))
        ));
    }

    #[test]
    fn test_valid_checksum() {
        // "abandon" x11 + "about" is a valid 12-word mnemonic with correct checksum
        let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
        assert!(validate_mnemonic(mnemonic).is_ok());
    }

    #[test]
    fn test_invalid_checksum() {
        // Swap the last word to corrupt the checksum (keeping all words valid)
        let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon";
        assert!(matches!(
            validate_mnemonic(mnemonic),
            Err(Bip39Error::InvalidChecksum)
        ));
    }

    #[test]
    fn test_valid_24_word_mnemonic() {
        // Standard 24-word BIP39 test vector
        let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon art";
        assert!(validate_mnemonic(mnemonic).is_ok());
    }

    #[test]
    fn test_seed_is_zeroizing() {
        let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
        let seed: Zeroizing<[u8; 64]> = mnemonic_to_seed(mnemonic, "").unwrap();
        assert_eq!(seed.len(), 64);
        assert_ne!(&seed[..16], &[0u8; 16][..]);
    }
}
