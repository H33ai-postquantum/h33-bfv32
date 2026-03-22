# H33 BFV32 — Open Source Fully Homomorphic Encryption

**Compute on encrypted data without ever decrypting it.**

32-bit BFV (Brakerski/Fan-Vercauteren) fully homomorphic encryption, optimized for ARM NEON. Open sourced by [H33.ai](https://h33.ai).

## What is FHE?

Fully Homomorphic Encryption allows mathematical operations on encrypted data. The result, when decrypted, is identical to performing the same operation on plaintext. The server never sees your data.

## Quick Start

```rust
use h33_bfv32::{Bfv32Parameters, BfvContext32, KeyGenerator32, Encryptor32, Decryptor32, BatchEncoder32};

let params = Bfv32Parameters::default_2048();
let ctx = BfvContext32::new(params);
let mut keygen = KeyGenerator32::new(&ctx);
let sk = keygen.generate_secret_key();
let pk = keygen.generate_public_key(&sk);

let encoder = BatchEncoder32::new(&ctx);
let encryptor = Encryptor32::new(&ctx, &pk);
let decryptor = Decryptor32::new(&ctx, &sk);

// Encrypt
let data = vec![42u32; ctx.params.n];
let encoded = encoder.encode(&data);
let encrypted = encryptor.encrypt(&encoded);

// Decrypt — result matches plaintext
let decrypted = decryptor.decrypt(&encrypted);
let decoded = encoder.decode(&decrypted);
assert_eq!(decoded[0], 42);
```

## Features

- BFV fully homomorphic encryption (32-bit coefficients)
- Montgomery NTT for fast polynomial multiplication
- ARM NEON SIMD acceleration
- Keygen, encrypt, decrypt, add, multiply, relinearize
- Batch encoding for SIMD parallelism
- CBD(3) error sampling

## Security

This library provides ~86-bit lattice security at N=2048. This is suitable for development, prototyping, and non-critical applications.

**For production post-quantum security (NIST Level 1+), use the full H33 platform:**

- [H33-128](https://h33.ai/h33-128) — 128-bit, NIST Level 1, 2.17M auth/sec
- [H33-CKKS](https://h33.ai/h33-ckks) — Encrypted float arithmetic for ML/AI
- [H33-256](https://h33.ai/h33-256) — 256-bit, NIST Level 5, maximum security
- [H33 FHE-IQ](https://h33.ai/fhe-iq) — Intelligent routing across all engines

## Why Open Source This?

H33's competitive advantage is in our post-quantum engines (H33-128, H33-CKKS, H33-256), our FHE-IQ intelligent router, and our production-scale optimizations that achieve 38.5µs per authentication. BFV32 is our mobile/edge engine — useful, correct, but not where our moat lives. We open-sourced it to help developers learn FHE and build prototypes.

## License

MIT

---

Built by [H33.ai](https://h33.ai) — Post-Quantum Encryption Platform
