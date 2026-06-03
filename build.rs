//! Build script: compile CUDA kernels to PTX when the `cuda` feature is on.
//!
//! Produces `$OUT_DIR/hash160.ptx`; the path is exposed to the crate via the
//! `KERNEL_HASH160_PTX` env var (read with `env!` in the host glue). No-op
//! unless `--features cuda` is set, so default/`cuda-sim` builds need no nvcc.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let cuda_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();
    if !cuda_enabled {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");

    // RTX 3070 Laptop is sm_86; overridable for other GPUs.
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_86".to_string());

    // (source stem, env var name exposed to the crate)
    let kernels = [
        ("hash160", "KERNEL_HASH160_PTX"),
        ("secp256k1", "KERNEL_SECP256K1_PTX"),
        ("derive", "KERNEL_DERIVE_PTX"),
    ];
    // Headers shared across kernels — recompile if they change.
    for h in ["secp256k1.cuh", "hash.cuh"] {
        println!(
            "cargo:rerun-if-changed={}",
            manifest_dir.join("kernels").join(h).display()
        );
    }

    for (stem, env_name) in kernels {
        let src = manifest_dir.join("kernels").join(format!("{stem}.cu"));
        let ptx = out_dir.join(format!("{stem}.ptx"));
        println!("cargo:rerun-if-changed={}", src.display());

        let status = Command::new("nvcc")
            .args(["-ptx", "-O3", &format!("-arch={arch}")])
            .arg(&src)
            .arg("-o")
            .arg(&ptx)
            .status()
            .expect("failed to invoke nvcc — is the CUDA toolkit on PATH?");

        if !status.success() {
            panic!("nvcc failed to compile {}", src.display());
        }
        println!("cargo:rustc-env={env_name}={}", ptx.display());
    }
}
