# blind-derivation

*[English](README.md)*

**GPU 加速的比特币 HD 钱包地址推导 —— 私钥永不离开本地。**

`blind-derivation` 把 HD 钱包地址生成拆成两个阶段,让繁重、可并行的部分能安全地外包给 GPU(本地或远程),**全程不暴露任何密钥材料**:

1. **本地 / 安全** —— 助记词 → 种子 → `xpub`(私钥留在本地)。
2. **GPU / 可外包** —— `xpub` → 数百万个子地址,全部在 GPU 上算。

因为非硬化 BIP32 派生只需要公钥和链码,GPU 是"盲"的:它能生成每一个地址,却永远无法反推出私钥。这就是 blind-derivation 里 "blind" 的含义。

> ⚡ 真实、实测的 GPU 推导 —— **RTX 3070 Laptop 上 ~2.0–2.7M 地址/秒**,约为 16 线程 CPU 的 16–20 倍。不是模拟。

---

## 为什么

生成大段地址区间(交易所充值墙、观察钱包、gap-limit 扫描、靓号搜索、链上分析)是高度可并行的,但又涉及密钥安全。常规做法会逼你做一个糟糕的取舍:要么在可信机器上慢慢算,要么把密钥材料送到你并不完全信任的快硬件上。

本项目消除了这个取舍。涉密步骤(PBKDF2 + 硬化 BIP32)在本地完成,只导出一个 `xpub`。之后的全部计算 —— HMAC-SHA512、secp256k1 点运算、HASH160 —— 仅凭公开数据在 GPU 上完成。

---

## 特性

- **两阶段安全模型** —— 助记词/种子/xprv 永不离开本地进程,只导出 `xpub` + 链码。
- **真正的 CUDA kernel** —— secp256k1 域/群运算、定基 `k·G`、HMAC-SHA512、SHA-256、RIPEMD-160 全部在设备端实现。GPU 路径上没有任何 CPU 回退。
- **逐字节正确性** —— 每个 GPU 结果都在测试套件里与纯 Rust 的 CPU 参照(`k256` + 本项目自带的 BIP32)对拍。
- **三种构建模式** —— 真 GPU(`cuda`)、无 GPU 开发用的 CPU 模拟(`cuda-sim`)、纯 CPU(默认)。
- **地址类型** —— P2PKH(传统)、P2WPKH(原生 SegWit / bech32)、P2SH-P2WPKH(包装 SegWit)。
- **CLI + 库** —— 既可当二进制用,也可作为 crate 嵌入。

---

## 性能

实测于 NVIDIA RTX 3070 Laptop(sm_86),release 构建,端到端含主机↔设备传输:

| 路径 | 吞吐 | 相对 CPU |
|------|-----:|--------:|
| CPU 基线(16 线程,rayon) | ~130k 地址/秒 | 1× |
| GPU,朴素 double-and-add | ~302k 地址/秒 | ~2.4× |
| **GPU,定基预计算表**(当前) | **~2.0–2.7M 地址/秒** | **~16–20×** |

定基窗口表把每个标量的 256 次倍点替换为 ≤32 次点加。如今每点的模逆(用于仿射转换)成了主要开销 —— 见[路线图](#路线图)。

---

## 安全模型

```
┌─────────────────────── 本地(安全)───────────────────────┐
│  助记词 ──PBKDF2──> 种子 ──BIP32──> 主 xprv                  │
│                                       │                     │
│                                       ▼  硬化路径            │
│                               账户 xprv                     │
│                                       │                     │
│                                       ▼  public_key()       │
│                               xpub + 链码  ──────────────┐  │
└───────────────────────────────────────────────────────  │ ─┘
                                                           │ 导出(仅公开数据)
┌─────────────────────── GPU(可不可信)──────────────────  ▼ ─┐
│  对区间内每个 index i:                                       │
│    I      = HMAC-SHA512(链码, xpub_pub || i)                 │
│    childₚ = xpub_pub + IL·G          (非硬化,公钥侧)        │
│    addr   = encode(HASH160(childₚ))                          │
└─────────────────────────────────────────────────────────────┘
```

- 助记词、种子、私钥**永不**跨越本地/GPU 边界。
- GPU 上只做非硬化派生,从 `xpub` + 子公钥反推私钥在数学上不可能。
- GPU 可以是远程租用的机器 —— 它只会看到公开数据。

> ⚠️ **尚未审计。** 这是研究级软件。BIP39 校验和未做验证(只查了词表成员),代码也未经安全审查。未经你自己审查前,请勿用于托管资金。见[注意事项](#注意事项)。

---

## 快速上手

### 环境要求

- Rust(stable,edition 2021)
- GPU 路径需要:CUDA Toolkit 12.x、`nvcc` 在 `PATH` 上、一块 NVIDIA GPU(sm_70+)。构建默认目标 `sm_86`,可用 `CUDA_ARCH` 覆盖(如 `CUDA_ARCH=sm_89`)。

### 构建

```bash
# 纯 CPU(无需 GPU)
cargo build --release

# 真 GPU(需要 CUDA 工具链 + NVIDIA GPU)
cargo build --release --features cuda

# GPU API 的 CPU 模拟(用于无 GPU 开发)
cargo build --release --features cuda-sim
```

### 命令行

```bash
# 1. 本地:从助记词推导 xpub(助记词务必保密!)
cargo run --release -- derive-xpub \
    --mnemonic "abandon abandon ... about" \
    --path "m/84'/0'/0'"

# 2. 从 xpub 批量推导地址(可安全外包)
cargo run --release -- batch-derive --xpub xpub6... --start 0 --count 1000

# 3. 端到端 demo:助记词 → xpub → 地址
cargo run --release -- demo --count 20

# 4. CPU 基准测试
cargo run --release -- benchmark --count 100000

# 5. 列出 CUDA 设备
cargo run --release --features cuda -- gpu-list

# 6. 真 GPU demo + 计时(对比 CPU)
cargo run --release --features cuda -- gpu-demo --count 200000
```

### 作为库

```rust
use blind_derivation::{bip32::ExtendedPrivateKey, bip39, batch, address::AddressType};

// 本地:助记词 -> xpub
let seed = bip39::mnemonic_to_seed("abandon abandon ... about", "")?;
let xpub = ExtendedPrivateKey::from_seed(&seed)?
    .derive_path("m/84'/0'/0'")?
    .public_key();

// 可外包:xpub -> 地址(此处为 CPU 基线;GPU 走 `cuda` feature)
let cfg = batch::BatchConfig { start_index: 0, count: 1000,
    address_type: AddressType::P2WPKH, mainnet: true };
let addresses = batch::batch_derive_cpu(&xpub, &cfg);
```

---

## 工作原理

### 阶段一 —— 本地(`bip39`、`bip32`)
- `bip39::mnemonic_to_seed` —— PBKDF2-HMAC-SHA512(2048 次迭代)。
- `ExtendedPrivateKey::from_seed` → `derive_path("m/84'/0'/0'")` → `public_key()` → `ExtendedPublicKey`。

### 阶段二 —— GPU(`kernels/`、`gpu_kernel`、`cuda`)
- `derive_child_kernel`(CUDA)对每个 index 计算 `childₚ = parent_pub + IL·G`,其中 `I = HMAC-SHA512(链码, parent_pub‖index)`,再算 `HASH160(childₚ)`。
- 父点在整个 batch 内是常量,所以在主机端解压一次、以仿射 `(x, y)` 传入 —— kernel 不做模平方根。
- `IL·G` 使用预计算的**定基窗口表**(8 位窗口,32×255 个仿射点,~522 KB,启动时用 `k256` 生成一次),因此每个标量只需 ≤32 次点加、零次倍点。

### Kernel
| 文件 | 内容 |
|------|------|
| `kernels/hash.cuh` | SHA-256、RIPEMD-160、HASH160、SHA-512、HMAC-SHA512(设备端) |
| `kernels/secp256k1.cuh` | 8×u32 域运算 mod p、Jacobian 点加/倍、Fermat 求逆、定基 `k·G`、点压缩 |
| `kernels/derive.cu` | `derive_child_kernel` —— 完整的 BIP32 公钥侧派生 |
| `build.rs` | 用 `nvcc` 把 `.cu` 编成 PTX,运行时由 `cudarc` 加载 |

---

## 测试

```bash
cargo test                      # CPU
cargo test --features cuda      # + 真 GPU kernel(需要 GPU)
```

GPU 测试是与 CPU 实现及 `k256` 的逐字节差分对拍:
- `sha256` / `hash160` 对已知向量及 CPU `address::hash160`
- `k·G` 对 `k256`,覆盖一批 256 位标量
- 完整 `derive_child`(公钥**和** HASH160)对 CPU,512 个 index 一批

---

## 路线图

A–E 阶段(工具链、哈希、secp256k1、HMAC + 派生、定基表)已完成。后续性能工作,按预期收益大致排序:

- [ ] **批量模逆**(Montgomery's trick)—— 摊销每点的 Fermat 求逆,目前的主要开销。
- [ ] **HMAC pad 主机端预计算** —— 每次派生省掉 4 次 SHA-512 压缩中的 2 次。
- [ ] **降低每线程 local memory** —— 派生 kernel 真正的占用率瓶颈。
- [ ] **多流分块** 应对超大 count(重叠拷贝/计算)。
- [ ] BIP39 校验和验证;base58/bech32 留在主机端(开销小)即可,但值得标注。

这些应能逼近原计划的 10M+ 地址/秒目标。

---

## 注意事项

- **研究级、未审计。** 没有安全审查。信任它处理真实资产前请自行审查。
- **BIP39 校验和未验证** —— 检查了词表成员,但未校验熵的校验位。
- GPU 上只对**非硬化**路径派生地址(刻意设计 —— 这正是它对私钥保持"盲"的原因)。
- GPU 结果依赖正确的 CUDA/驱动配置;不一致会在测试套件里暴露,测试套件是正确性的唯一标准。

---

## 致谢

独立实现。GPU 方案参考了对现有开源 secp256k1/GPU 工作的调研 —— 尤其是 [CudaBrainSecp](https://github.com/XopMC/CudaBrainSecp)(MIT,定基表技术)和 [CUDA_Mnemonic_Recovery](https://github.com/XopMC/CUDA_Mnemonic_Recovery)(Apache-2.0)。此处所有 kernel 均为原创;正确性以 [`k256`](https://crates.io/crates/k256) crate 为锚点。

## 许可证

双协议授权,任选其一:

- Apache License, Version 2.0([LICENSE-APACHE](LICENSE-APACHE))
- MIT 协议([LICENSE-MIT](LICENSE-MIT))

除非你另行声明,你有意提交以纳入本作品的任何贡献(如 Apache-2.0 协议所定义),均按上述双协议授权,不附加任何额外条款。
