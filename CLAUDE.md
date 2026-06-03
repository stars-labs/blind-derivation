# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build (default, no CUDA)
cargo build

# Build with CUDA simulation (no GPU hardware required)
cargo build --release --features cuda-sim

# Build with real CUDA (requires CUDA 12.8 toolkit)
cargo build --release --features cuda

# Run all tests
cargo test

# Run a single test
cargo test test_mnemonic_to_seed_vector

# Run tests for a specific module
cargo test bip39::tests
cargo test bip32::tests

# Run benchmarks
cargo bench

# Run CLI
cargo run -- demo
cargo run -- derive-xpub --mnemonic "word1 word2 ..."
cargo run -- batch-derive --xpub <xpub_string> --count 100
cargo run -- benchmark --count 10000
cargo run -- gpu-list
cargo run --features cuda-sim -- gpu-demo --count 1000000
```

## Architecture

The crate implements a **two-phase HD wallet address derivation** model that separates secret key material from the batch address generation:

**Phase 1 — Local/Secure** (`bip39`, `bip32` modules):
- `bip39::mnemonic_to_seed` — PBKDF2-HMAC-SHA512 (2048 iterations) converts mnemonic → 64-byte seed
- `bip32::ExtendedPrivateKey::from_seed` → `derive_path("m/84'/0'/0'")` → `public_key()` extracts `ExtendedPublicKey`
- All private key material stays local; only the `xpub` is exported

**Phase 2 — Remote/GPU-acceleratable** (`batch`, `cuda` modules):
- `ExtendedPublicKey::derive_child(index)` — non-hardened BIP32 child derivation using HMAC-SHA512 + secp256k1 point addition
- `batch::batch_derive_cpu` parallelizes this with rayon across CPU cores
- `cuda::CudaContext::batch_derive` is the GPU path (cudarc 0.18); the actual PTX kernel is not yet implemented (TODO in `cuda.rs:218`)
- `address` module converts 33-byte compressed public keys → P2PKH / P2WPKH (bech32) / P2SH-P2WPKH addresses

**Feature flags** control the CUDA backend:
- `cuda` — real GPU (requires CUDA toolkit at build time)
- `cuda-sim` — simulation mode with fake device stats and timing
- Neither — stub that returns `CudaError::NotAvailable`

**Key types and their flow**:
```
mnemonic → [u8; 64] seed → ExtendedPrivateKey → ExtendedPublicKey
                                                    ↓
                                             GpuExportData ([u8; 65])
                                                    ↓
                                             DerivedAddress { index, address, pubkey }
```

## Notes

- The BIP39 checksum validation is **not implemented** (`bip39.rs:41`) — the wordlist membership check is done but entropy-bits checksum is skipped
- The CUDA kernel PTX is a stub (`cuda.rs:218`); real GPU derivation requires implementing secp256k1 point addition in CUDA
- Address serialization in `main.rs::parse_xpub` manually decodes the base58check bytes (bytes 4–78); this is separate from `ExtendedPublicKey::to_bytes`/`to_base58` which always writes mainnet version bytes (`0x0488B21E`)
