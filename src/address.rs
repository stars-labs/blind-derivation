//! Bitcoin Address Generation
//!
//! Supports P2PKH (legacy), P2WPKH (native segwit), and P2SH-P2WPKH (wrapped segwit)

use ripemd::Ripemd160;
use sha2::{Digest, Sha256};

/// Compute HASH160 (SHA256 followed by RIPEMD160)
pub fn hash160(data: &[u8]) -> [u8; 20] {
    let sha256_hash = Sha256::digest(data);
    let ripemd_hash = Ripemd160::digest(sha256_hash);
    let mut result = [0u8; 20];
    result.copy_from_slice(&ripemd_hash);
    result
}

/// Generate P2PKH address (legacy, starts with 1)
pub fn p2pkh_address(pubkey: &[u8; 33], mainnet: bool) -> String {
    let pubkey_hash = hash160(pubkey);

    // Version byte: 0x00 for mainnet, 0x6f for testnet
    let version = if mainnet { 0x00 } else { 0x6f };

    let mut payload = vec![version];
    payload.extend_from_slice(&pubkey_hash);

    bs58::encode(payload).with_check().into_string()
}

/// Generate P2WPKH address (native segwit, starts with bc1)
pub fn p2wpkh_address(pubkey: &[u8; 33], mainnet: bool) -> String {
    let pubkey_hash = hash160(pubkey);

    let hrp = if mainnet {
        bech32::hrp::BC
    } else {
        bech32::hrp::TB
    };

    // Use segwit module for proper encoding
    bech32::segwit::encode(hrp, bech32::segwit::VERSION_0, &pubkey_hash)
        .expect("valid segwit address")
}

/// Generate P2SH-P2WPKH address (wrapped segwit, starts with 3)
pub fn p2sh_p2wpkh_address(pubkey: &[u8; 33], mainnet: bool) -> String {
    let pubkey_hash = hash160(pubkey);

    // Create witness script: OP_0 <20-byte-pubkey-hash>
    let mut witness_script = vec![0x00, 0x14]; // OP_0, PUSH20
    witness_script.extend_from_slice(&pubkey_hash);

    // Hash the witness script
    let script_hash = hash160(&witness_script);

    // Version byte: 0x05 for mainnet P2SH, 0xc4 for testnet
    let version = if mainnet { 0x05 } else { 0xc4 };

    let mut payload = vec![version];
    payload.extend_from_slice(&script_hash);

    bs58::encode(payload).with_check().into_string()
}


/// Address type enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressType {
    P2PKH,      // Legacy (1...)
    P2WPKH,     // Native SegWit (bc1q...)
    P2SHP2WPKH, // Wrapped SegWit (3...)
}

/// Generate address of specified type
pub fn generate_address(pubkey: &[u8; 33], address_type: AddressType, mainnet: bool) -> String {
    match address_type {
        AddressType::P2PKH => p2pkh_address(pubkey, mainnet),
        AddressType::P2WPKH => p2wpkh_address(pubkey, mainnet),
        AddressType::P2SHP2WPKH => p2sh_p2wpkh_address(pubkey, mainnet),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash160() {
        // Test vector
        let pubkey = hex::decode("0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798")
            .unwrap();
        let hash = hash160(&pubkey);
        let expected = hex::decode("751e76e8199196d454941c45d1b3a323f1433bd6").unwrap();
        assert_eq!(hash.to_vec(), expected);
    }

    #[test]
    fn test_p2pkh_address() {
        // Generator point public key (compressed)
        let pubkey: [u8; 33] = hex::decode("0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798")
            .unwrap()
            .try_into()
            .unwrap();

        let address = p2pkh_address(&pubkey, true);
        assert_eq!(address, "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH");
    }

    #[test]
    fn test_p2wpkh_address() {
        let pubkey: [u8; 33] = hex::decode("0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798")
            .unwrap()
            .try_into()
            .unwrap();

        let address = p2wpkh_address(&pubkey, true);
        assert_eq!(address, "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4");
    }

    #[test]
    fn test_p2sh_p2wpkh_address() {
        let pubkey: [u8; 33] = hex::decode("0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798")
            .unwrap()
            .try_into()
            .unwrap();

        let address = p2sh_p2wpkh_address(&pubkey, true);
        // This should be a valid P2SH address starting with 3
        assert!(address.starts_with('3'));
    }
}
