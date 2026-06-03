//! Benchmarks for address derivation performance

use blind_derivation::{
    address::AddressType,
    batch::{batch_derive_cpu, batch_derive_sequential, BatchConfig},
    bip32::ExtendedPrivateKey,
    bip39::mnemonic_to_seed,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn setup_xpub() -> blind_derivation::ExtendedPublicKey {
    let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
    let seed = mnemonic_to_seed(mnemonic, "").unwrap();
    let master = ExtendedPrivateKey::from_seed(&*seed).unwrap();
    let account = master.derive_path("m/84'/0'/0'").unwrap();
    let xpub = account.public_key();
    xpub.derive_path("m/0").unwrap()
}

fn bench_single_derivation(c: &mut Criterion) {
    let xpub = setup_xpub();

    c.bench_function("single_address_derivation", |b| {
        b.iter(|| {
            let child = xpub.derive_child(black_box(0)).unwrap();
            blind_derivation::address::p2wpkh_address(&child.key, true)
        })
    });
}

fn bench_batch_derivation(c: &mut Criterion) {
    let xpub = setup_xpub();

    let mut group = c.benchmark_group("batch_derivation");

    for count in [100, 1000, 10000].iter() {
        let config = BatchConfig {
            start_index: 0,
            count: *count,
            address_type: AddressType::P2WPKH,
            mainnet: true,
        };

        group.bench_with_input(
            BenchmarkId::new("parallel", count),
            count,
            |b, _| {
                b.iter(|| batch_derive_cpu(black_box(&xpub), black_box(&config)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sequential", count),
            count,
            |b, _| {
                b.iter(|| batch_derive_sequential(black_box(&xpub), black_box(&config)))
            },
        );
    }

    group.finish();
}

fn bench_address_types(c: &mut Criterion) {
    let xpub = setup_xpub();
    let child = xpub.derive_child(0).unwrap();

    let mut group = c.benchmark_group("address_encoding");

    group.bench_function("p2pkh", |b| {
        b.iter(|| blind_derivation::address::p2pkh_address(black_box(&child.key), true))
    });

    group.bench_function("p2wpkh", |b| {
        b.iter(|| blind_derivation::address::p2wpkh_address(black_box(&child.key), true))
    });

    group.bench_function("p2sh_p2wpkh", |b| {
        b.iter(|| blind_derivation::address::p2sh_p2wpkh_address(black_box(&child.key), true))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_derivation,
    bench_batch_derivation,
    bench_address_types
);
criterion_main!(benches);
