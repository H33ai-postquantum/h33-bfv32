//! # H33 BFV32 — Open Source 32-bit Fully Homomorphic Encryption
//!
//! A fully homomorphic encryption library using the BFV scheme with 32-bit
//! coefficients, optimized for ARM NEON. Compute on encrypted data without
//! ever decrypting it.
//!
//! ## Quick Start
//!
//! ```rust
//! use h33_bfv32::{Bfv32Parameters, BfvContext32, KeyGenerator32, Encryptor32, Decryptor32, BatchEncoder32};
//!
//! let params = Bfv32Parameters::default_2048();
//! let ctx = BfvContext32::new(params);
//! let mut keygen = KeyGenerator32::new(&ctx);
//! let sk = keygen.generate_secret_key();
//! let pk = keygen.generate_public_key(&sk);
//!
//! let encoder = BatchEncoder32::new(&ctx);
//! let encryptor = Encryptor32::new(&ctx, &pk);
//! let decryptor = Decryptor32::new(&ctx, &sk);
//!
//! let plaintext = vec![42u32; ctx.params.n];
//! let encoded = encoder.encode(&plaintext);
//! let encrypted = encryptor.encrypt(&encoded);
//! let decrypted = decryptor.decrypt(&encrypted);
//! let decoded = encoder.decode(&decrypted);
//!
//! assert_eq!(decoded[0], 42);
//! ```
//!
//! ## Features
//!
//! - BFV fully homomorphic encryption with 32-bit coefficients
//! - Montgomery NTT for fast polynomial multiplication
//! - ARM NEON SIMD acceleration (falls back to scalar on x86)
//! - Keygen, encrypt, decrypt, add, multiply operations
//! - Batch encoding for SIMD parallelism
//!
//! ## Security Note
//!
//! This library provides ~86-bit lattice security at N=2048. For NIST
//! Level 1+ post-quantum security, use the full H33 platform at
//! [h33.ai](https://h33.ai) which provides H33-128 (128-bit, N=4096)
//! and H33-256 (256-bit, N=16384).
//!
//! ## License
//!
//! MIT — see LICENSE file.
//!
//! Built by [H33.ai](https://h33.ai) — Post-Quantum Encryption Platform

pub mod bfv32;
pub mod ntt32;
pub mod butterfly;

pub use bfv32::*;
pub use ntt32::*;
