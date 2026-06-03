//! Blind Derivation CLI Demo
//!
//! Demonstrates the HD wallet batch address derivation workflow:
//! 1. Local: Mnemonic → Seed → xpub
//! 2. Remote-safe: xpub → Batch addresses

use blind_derivation::{
    address::AddressType,
    batch::{batch_derive_cpu, BatchConfig, GpuExportData},
    bip32::ExtendedPrivateKey,
    bip39::mnemonic_to_seed,
    cuda::{CudaBatchConfig, CudaContext},
};
use clap::{Parser, Subcommand};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "blind-derivation")]
#[command(about = "HD Wallet Batch Address Derivation with GPU support")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Derive xpub from mnemonic (LOCAL ONLY - keep mnemonic secure!)
    DeriveXpub {
        /// BIP39 mnemonic (24 words)
        #[arg(short, long)]
        mnemonic: String,

        /// Optional passphrase
        #[arg(short, long, default_value = "")]
        passphrase: String,

        /// Derivation path (default: m/84'/0'/0')
        #[arg(short, long, default_value = "m/84'/0'/0'")]
        path: String,
    },

    /// Batch derive addresses from xpub (SAFE TO OUTSOURCE)
    BatchDerive {
        /// Extended public key (xpub)
        #[arg(short, long)]
        xpub: String,

        /// Starting index
        #[arg(short, long, default_value = "0")]
        start: u32,

        /// Number of addresses to derive
        #[arg(short, long, default_value = "10")]
        count: u32,

        /// Address type: p2pkh, p2wpkh, p2sh-p2wpkh
        #[arg(short, long, default_value = "p2wpkh")]
        address_type: String,

        /// Use testnet
        #[arg(short = 't', long)]
        testnet: bool,
    },

    /// Full demo: mnemonic → xpub → batch addresses
    Demo {
        /// Number of addresses to derive
        #[arg(short, long, default_value = "10")]
        count: u32,
    },

    /// Benchmark batch derivation performance
    Benchmark {
        /// Number of addresses to derive
        #[arg(short, long, default_value = "10000")]
        count: u32,
    },

    /// Export xpub data for GPU processing
    ExportGpu {
        /// Extended public key (xpub)
        #[arg(short, long)]
        xpub: String,

        /// Output format: hex, base64
        #[arg(short, long, default_value = "hex")]
        format: String,
    },

    /// GPU demo: test CUDA and run batch derivation
    GpuDemo {
        /// Number of addresses to derive
        #[arg(short, long, default_value = "1000000")]
        count: u32,

        /// GPU device ID
        #[arg(short, long, default_value = "0")]
        device: usize,
    },

    /// List available CUDA devices
    GpuList,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::DeriveXpub {
            mnemonic,
            passphrase,
            path,
        } => {
            derive_xpub(&mnemonic, &passphrase, &path);
        }
        Commands::BatchDerive {
            xpub,
            start,
            count,
            address_type,
            testnet,
        } => {
            batch_derive(&xpub, start, count, &address_type, !testnet);
        }
        Commands::Demo { count } => {
            run_demo(count);
        }
        Commands::Benchmark { count } => {
            run_benchmark(count);
        }
        Commands::ExportGpu { xpub, format } => {
            export_gpu(&xpub, &format);
        }
        Commands::GpuDemo { count, device } => {
            run_gpu_demo(count, device);
        }
        Commands::GpuList => {
            list_gpu_devices();
        }
    }
}

fn derive_xpub(mnemonic: &str, passphrase: &str, path: &str) {
    println!("=== Deriving xpub from mnemonic (LOCAL ONLY) ===\n");

    // Step 1: Mnemonic to seed
    println!("Step 1: Converting mnemonic to seed...");
    let seed = match mnemonic_to_seed(mnemonic, passphrase) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: {}", e);
            return;
        }
    };
    println!("  Seed derived (KEEP SECRET!)\n");

    // Step 2: Master key
    println!("Step 2: Deriving master key...");
    let master = match ExtendedPrivateKey::from_seed(&*seed) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error: {}", e);
            return;
        }
    };
    println!("  Master xprv: {}...", &master.to_base58()[..20]);
    println!("  (KEEP SECRET!)\n");

    // Step 3: Derive account
    println!("Step 3: Deriving account at path: {}", path);
    let account = match master.derive_path(path) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error: {}", e);
            return;
        }
    };
    let xpub = account.public_key();

    println!("\n=== RESULT ===\n");
    println!("xpub (SAFE TO SHARE for non-hardened derivation):");
    println!("{}\n", xpub.to_base58());

    println!("Chain code (hex):");
    println!("{}\n", hex::encode(xpub.chain_code));

    println!("Public key (hex):");
    println!("{}\n", hex::encode(xpub.key));
}

fn batch_derive(xpub_str: &str, start: u32, count: u32, address_type: &str, mainnet: bool) {
    println!("=== Batch Deriving Addresses (SAFE TO OUTSOURCE) ===\n");

    // Parse xpub
    let xpub_bytes = match bs58::decode(xpub_str).with_check(None).into_vec() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Error decoding xpub: {}", e);
            return;
        }
    };

    if xpub_bytes.len() != 78 {
        eprintln!("Invalid xpub length");
        return;
    }

    let xpub = parse_xpub(&xpub_bytes);

    let addr_type = match address_type.to_lowercase().as_str() {
        "p2pkh" | "legacy" => AddressType::P2PKH,
        "p2wpkh" | "native" | "bech32" => AddressType::P2WPKH,
        "p2sh-p2wpkh" | "wrapped" => AddressType::P2SHP2WPKH,
        _ => {
            eprintln!("Unknown address type: {}", address_type);
            return;
        }
    };

    let config = BatchConfig {
        start_index: start,
        count,
        address_type: addr_type,
        mainnet,
    };

    println!(
        "Deriving {} addresses starting from index {}...\n",
        count, start
    );

    let start_time = Instant::now();
    let results = batch_derive_cpu(&xpub, &config);
    let elapsed = start_time.elapsed();

    println!("Index\tAddress");
    println!("{}", "-".repeat(60));

    for result in results {
        match result {
            Ok(addr) => println!("{}\t{}", addr.index, addr.address),
            Err(e) => eprintln!("Error: {}", e),
        }
    }

    println!("\nDerived {} addresses in {:?}", count, elapsed);
}

fn run_demo(count: u32) {
    println!("=== BLIND DERIVATION DEMO ===\n");
    println!("This demo shows the complete workflow:\n");
    println!("1. LOCAL (secure): Mnemonic → Seed → xpub");
    println!("2. REMOTE-SAFE: xpub → Batch addresses\n");
    println!("{}\n", "=".repeat(50));

    // Use a well-known test mnemonic
    let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";

    println!("TEST MNEMONIC (DO NOT USE IN PRODUCTION!):");
    println!("{}\n", mnemonic);

    // Phase 1: Local derivation
    println!("--- Phase 1: LOCAL DERIVATION ---\n");

    let seed = mnemonic_to_seed(mnemonic, "").expect("valid mnemonic");
    println!("Seed (first 16 bytes): {}...", hex::encode(&seed[..16]));

    let master = ExtendedPrivateKey::from_seed(&*seed).expect("valid seed");
    println!("Master xprv: {}...", &master.to_base58()[..30]);

    // Derive BIP84 account (native segwit)
    let account = master.derive_path("m/84'/0'/0'").expect("valid path");
    let xpub = account.public_key();

    println!("\nAccount xpub (m/84'/0'/0'):");
    println!("{}\n", xpub.to_base58());

    println!("This xpub is SAFE to share! Remote servers can derive addresses");
    println!("but CANNOT derive private keys.\n");

    // Phase 2: Batch derivation
    println!("--- Phase 2: BATCH DERIVATION (GPU-compatible) ---\n");

    let config = BatchConfig {
        start_index: 0,
        count,
        address_type: AddressType::P2WPKH,
        mainnet: true,
    };

    let start_time = Instant::now();
    let _results = batch_derive_cpu(&xpub, &config);
    let elapsed = start_time.elapsed();

    println!("Derived {} addresses in {:?}\n", count, elapsed);

    println!("First 10 receive addresses (m/84'/0'/0'/0/i):");
    println!("{}", "-".repeat(50));

    // Derive external chain (m/84'/0'/0'/0)
    let external = xpub.derive_path("m/0").expect("valid path");
    let external_config = BatchConfig {
        start_index: 0,
        count: count.min(10),
        address_type: AddressType::P2WPKH,
        mainnet: true,
    };

    let external_addrs = batch_derive_cpu(&external, &external_config);
    for addr in external_addrs.into_iter().flatten() {
        println!("  m/84'/0'/0'/0/{}: {}", addr.index, addr.address);
    }

    println!("\nFirst 5 change addresses (m/84'/0'/0'/1/i):");
    println!("{}", "-".repeat(50));

    // Derive internal (change) chain (m/84'/0'/0'/1)
    let internal = xpub.derive_path("m/1").expect("valid path");
    let internal_config = BatchConfig {
        start_index: 0,
        count: 5,
        address_type: AddressType::P2WPKH,
        mainnet: true,
    };

    let internal_addrs = batch_derive_cpu(&internal, &internal_config);
    for addr in internal_addrs.into_iter().flatten() {
        println!("  m/84'/0'/0'/1/{}: {}", addr.index, addr.address);
    }

    println!("\n=== SECURITY SUMMARY ===\n");
    println!("✓ Mnemonic never leaves local machine");
    println!("✓ Seed never leaves local machine");
    println!("✓ xprv (private key) never leaves local machine");
    println!("✓ Only xpub exported for batch derivation");
    println!("✓ Cannot reverse xpub to get private keys");
    println!("✓ GPU can batch derive addresses from xpub safely");
}

fn run_benchmark(count: u32) {
    println!("=== BENCHMARK: Batch Address Derivation ===\n");

    // Setup
    let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
    let seed = mnemonic_to_seed(mnemonic, "").expect("valid mnemonic");
    let master = ExtendedPrivateKey::from_seed(&*seed).expect("valid seed");
    let account = master.derive_path("m/84'/0'/0'").expect("valid path");
    let xpub = account.public_key();
    let external = xpub.derive_path("m/0").expect("valid path");

    println!("Configuration:");
    println!("  Addresses to derive: {}", count);
    println!("  Address type: P2WPKH (native segwit)");
    println!(
        "  CPU threads: {} (rayon auto)",
        rayon::current_num_threads()
    );
    println!();

    let config = BatchConfig {
        start_index: 0,
        count,
        address_type: AddressType::P2WPKH,
        mainnet: true,
    };

    // Warm up
    println!("Warming up...");
    let _ = batch_derive_cpu(
        &external,
        &BatchConfig {
            count: 100,
            ..config.clone()
        },
    );

    // Benchmark
    println!("Running benchmark...\n");

    let start = Instant::now();
    let results = batch_derive_cpu(&external, &config);
    let elapsed = start.elapsed();

    let successful = results.iter().filter(|r| r.is_ok()).count();
    let rate = count as f64 / elapsed.as_secs_f64();

    println!("=== RESULTS ===\n");
    println!("Total addresses:    {}", count);
    println!("Successful:         {}", successful);
    println!("Time elapsed:       {:?}", elapsed);
    println!("Derivation rate:    {:.2} addresses/second", rate);
    println!();
    println!(
        "Estimated time for 100M addresses: {:.2} seconds",
        100_000_000.0 / rate
    );
    println!();
    println!("Note: GPU (CUDA) implementation can achieve 10-100x speedup.");
}

fn export_gpu(xpub_str: &str, format: &str) {
    println!("=== Export xpub for GPU Processing ===\n");

    let xpub_bytes = match bs58::decode(xpub_str).with_check(None).into_vec() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Error decoding xpub: {}", e);
            return;
        }
    };

    if xpub_bytes.len() != 78 {
        eprintln!("Invalid xpub length");
        return;
    }

    let xpub = parse_xpub(&xpub_bytes);
    let export_data: GpuExportData = (&xpub).into();
    let bytes = export_data.to_bytes();

    println!("GPU Export Data (65 bytes):\n");

    match format.to_lowercase().as_str() {
        "hex" => {
            println!("Hex:");
            println!("{}\n", hex::encode(bytes));
        }
        "base64" => {
            use std::io::Write;
            let mut buf = Vec::new();
            write!(buf, "{}", base64_encode(&bytes)).unwrap();
            println!("Base64:");
            println!("{}\n", String::from_utf8(buf).unwrap());
        }
        _ => {
            eprintln!("Unknown format: {}", format);
            return;
        }
    }

    println!("Components:");
    println!(
        "  Public key (33 bytes): {}",
        hex::encode(export_data.pubkey)
    );
    println!(
        "  Chain code (32 bytes): {}",
        hex::encode(export_data.chain_code)
    );
    println!();
    println!("This data is sufficient for GPU to derive unlimited non-hardened addresses.");
    println!("Private keys CANNOT be derived from this data.");
}

fn parse_xpub(bytes: &[u8]) -> blind_derivation::ExtendedPublicKey {
    let depth = bytes[4];
    let mut parent_fingerprint = [0u8; 4];
    parent_fingerprint.copy_from_slice(&bytes[5..9]);
    let child_number = u32::from_be_bytes(bytes[9..13].try_into().unwrap());
    let mut chain_code = [0u8; 32];
    chain_code.copy_from_slice(&bytes[13..45]);
    let mut key = [0u8; 33];
    key.copy_from_slice(&bytes[45..78]);

    blind_derivation::ExtendedPublicKey {
        key,
        chain_code,
        depth,
        parent_fingerprint,
        child_number,
    }
}

fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();

    for chunk in data.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = chunk.get(1).copied().unwrap_or(0) as usize;
        let b2 = chunk.get(2).copied().unwrap_or(0) as usize;

        result.push(ALPHABET[b0 >> 2] as char);
        result.push(ALPHABET[((b0 & 0x03) << 4) | (b1 >> 4)] as char);

        if chunk.len() > 1 {
            result.push(ALPHABET[((b1 & 0x0f) << 2) | (b2 >> 6)] as char);
        } else {
            result.push('=');
        }

        if chunk.len() > 2 {
            result.push(ALPHABET[b2 & 0x3f] as char);
        } else {
            result.push('=');
        }
    }

    result
}

fn list_gpu_devices() {
    println!("=== CUDA GPU Devices ===\n");

    match CudaContext::list_devices() {
        Ok(devices) => {
            if devices.is_empty() {
                println!("No CUDA devices found.");
            } else {
                for (i, dev) in devices.iter().enumerate() {
                    println!("Device {}: {}", i, dev.name);
                    println!(
                        "  Compute Capability: {}.{}",
                        dev.compute_capability.0, dev.compute_capability.1
                    );
                    println!("  Total Memory: {} MB", dev.total_memory / 1024 / 1024);
                    println!("  Multiprocessors: {}", dev.multiprocessor_count);
                    println!();
                }
            }
        }
        Err(e) => {
            eprintln!("Error listing devices: {}", e);
            eprintln!("\nHint: Build with CUDA support: cargo build --release --features cuda");
        }
    }
}

fn run_gpu_demo(count: u32, device_id: usize) {
    println!("=== GPU DEMO: CUDA Batch Address Derivation ===\n");

    // Step 1: Initialize CUDA
    println!("Step 1: Initializing CUDA device {}...", device_id);
    let cuda_ctx = match CudaContext::new(device_id) {
        Ok(ctx) => {
            println!("  Device: {}", ctx.device_info.name);
            println!(
                "  Compute: SM {}.{}",
                ctx.device_info.compute_capability.0, ctx.device_info.compute_capability.1
            );
            println!(
                "  Memory: {} MB",
                ctx.device_info.total_memory / 1024 / 1024
            );
            println!("  MPs: {}\n", ctx.device_info.multiprocessor_count);
            ctx
        }
        Err(e) => {
            eprintln!("Failed to initialize CUDA: {}\n", e);
            eprintln!("Hint: Build with CUDA support: cargo build --release --features cuda");
            return;
        }
    };

    // Step 2: Test GPU
    println!("Step 2: Testing GPU memory transfer...");
    match cuda_ctx.test_gpu() {
        Ok(msg) => println!("  {}\n", msg),
        Err(e) => {
            eprintln!("  GPU test failed: {}", e);
            return;
        }
    }

    // Step 3: Prepare xpub (using test mnemonic)
    println!("Step 3: Preparing xpub data (LOCAL)...");
    let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
    let seed = mnemonic_to_seed(mnemonic, "").expect("valid mnemonic");
    let master = ExtendedPrivateKey::from_seed(&*seed).expect("valid seed");
    let account = master.derive_path("m/84'/0'/0'").expect("valid path");
    let xpub = account.public_key();
    let external = xpub.derive_path("m/0").expect("valid path");

    println!("  xpub: {}...", &external.to_base58()[..40]);

    let gpu_data: GpuExportData = (&external).into();
    println!("  Exported {} bytes to GPU\n", gpu_data.to_bytes().len());

    // Step 4: Run GPU batch derivation
    println!(
        "Step 4: Running GPU batch derivation ({} addresses)...",
        count
    );

    let config = CudaBatchConfig {
        start_index: 0,
        count,
        block_size: 256,
        num_streams: 4,
    };

    match cuda_ctx.batch_derive(&gpu_data, &config) {
        Ok(result) => {
            let rate = result.count as f64 / (result.elapsed_ms / 1000.0);
            println!("\n=== GPU RESULTS ===\n");
            println!("Addresses derived: {}", result.count);
            println!("Time elapsed:      {:.2} ms", result.elapsed_ms);
            println!("Derivation rate:   {:.2} addresses/sec", rate);
            println!();
            println!(
                "Estimated time for 100M addresses: {:.2} seconds",
                100_000_000.0 / rate
            );
        }
        Err(e) => {
            eprintln!("GPU derivation failed: {}", e);
        }
    }

    // Step 5: Compare with CPU
    println!("\n--- CPU Comparison ---\n");

    let cpu_config = BatchConfig {
        start_index: 0,
        count: count.min(10000), // Limit CPU for fair comparison
        address_type: AddressType::P2WPKH,
        mainnet: true,
    };

    let start = Instant::now();
    let results = batch_derive_cpu(&external, &cpu_config);
    let elapsed = start.elapsed();

    let cpu_count = results.iter().filter(|r| r.is_ok()).count() as u32;
    let cpu_rate = cpu_count as f64 / elapsed.as_secs_f64();

    println!(
        "CPU ({} addresses): {:.2} ms ({:.2} addr/sec)",
        cpu_count,
        elapsed.as_secs_f64() * 1000.0,
        cpu_rate
    );
    println!();

    println!("=== SECURITY SUMMARY ===\n");
    println!("  Mnemonic:    NEVER left local machine");
    println!("  Seed:        NEVER left local machine");
    println!("  Private key: NEVER left local machine");
    println!("  xpub:        Safely exported to GPU");
    println!("  GPU cannot reverse-engineer private keys");
}
