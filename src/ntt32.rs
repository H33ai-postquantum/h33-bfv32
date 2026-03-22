//! 32-bit NTT for NEON optimization
//!
//! Uses u32 coefficients with moduli < 2^30.
//! On ARM, uses vmull_u32 for 4-lane SIMD butterfly operations.
//! On x86/other, falls back to scalar.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use crate::butterfly::{
    butterfly_ct_neon_4x32, butterfly_gs_neon_4x32,
    butterfly_ct_scalar_32, butterfly_gs_scalar_32,
};

/// Pre-computed 32-bit NTT tables
#[derive(Clone)]
pub struct Ntt32Tables {
    /// Polynomial degree
    n: usize,
    /// log2(n)
    log_n: usize,
    /// Bit-reversal permutation
    bit_rev_table: Vec<usize>,
    /// Forward twiddles in Montgomery form (u32)
    twiddles_mont: Vec<u32>,
    /// Inverse twiddles in Montgomery form (u32)
    inv_twiddles_mont: Vec<u32>,
    /// Modulus (< 2^30)
    q: u32,
    /// -q^(-1) mod 2^32
    q_inv_neg: u32,
    /// R^2 mod q
    r2_mod_q: u32,
    /// n^(-1) in Montgomery form
    n_inv_mont: u32,
}

impl Ntt32Tables {
    /// Precompute tables for 32-bit NTT
    pub fn new(n: usize, q: u32) -> Self {
        assert!(n.is_power_of_two());
        assert!(q < (1 << 30), "Modulus must be < 2^30 for 32-bit NTT");

        let log_n = n.trailing_zeros() as usize;

        // Bit-reversal table
        let bit_rev_table: Vec<usize> = (0..n)
            .map(|i| i.reverse_bits() >> (usize::BITS as usize - log_n))
            .collect();

        // Montgomery constants
        let q_inv_neg = Self::compute_q_inv_neg(q);
        let r2_mod_q = Self::compute_r2_mod_q(q);

        // Find primitive 2n-th root of unity
        let g = Self::primitive_root(q);
        let power = (q - 1) / (2 * n as u32);
        let psi = Self::mod_pow(g, power, q);
        let psi_inv = Self::mod_inverse(psi, q);

        // Compute twiddles and convert to Montgomery form
        let mut twiddles_mont = Vec::with_capacity(n);
        let mut inv_twiddles_mont = Vec::with_capacity(n);
        let mut w = 1u32;
        let mut w_inv = 1u32;

        for _ in 0..n {
            twiddles_mont.push(Self::to_montgomery(w, q, r2_mod_q, q_inv_neg));
            inv_twiddles_mont.push(Self::to_montgomery(w_inv, q, r2_mod_q, q_inv_neg));
            w = Self::mul_mod(w, psi, q);
            w_inv = Self::mul_mod(w_inv, psi_inv, q);
        }

        // n^(-1) in Montgomery form
        let n_inv = Self::mod_inverse(n as u32, q);
        let n_inv_mont = Self::to_montgomery(n_inv, q, r2_mod_q, q_inv_neg);

        Self {
            n,
            log_n,
            bit_rev_table,
            twiddles_mont,
            inv_twiddles_mont,
            q,
            q_inv_neg,
            r2_mod_q,
            n_inv_mont,
        }
    }

    fn compute_q_inv_neg(q: u32) -> u32 {
        let mut x = q;
        for _ in 0..5 {
            x = x.wrapping_mul(2u32.wrapping_sub(q.wrapping_mul(x)));
        }
        x.wrapping_neg()
    }

    fn compute_r2_mod_q(q: u32) -> u32 {
        let mut r = 1u32;
        for _ in 0..32 {
            r = ((r as u64) << 1).rem_euclid(q as u64) as u32;
        }
        ((r as u64 * r as u64) % q as u64) as u32
    }

    fn to_montgomery(a: u32, q: u32, r2_mod_q: u32, q_inv_neg: u32) -> u32 {
        let prod = a as u64 * r2_mod_q as u64;
        let t_lo = prod as u32;
        let m = t_lo.wrapping_mul(q_inv_neg);
        let mq = m as u64 * q as u64;
        let sum = prod.wrapping_add(mq);
        let result = (sum >> 32) as u32;
        if result >= q { result - q } else { result }
    }

    fn primitive_root(q: u32) -> u32 {
        for g in 2..q {
            if Self::is_primitive_root(g, q) {
                return g;
            }
        }
        panic!("No primitive root found");
    }

    fn is_primitive_root(g: u32, q: u32) -> bool {
        let phi = q - 1;
        let mut d = 2u32;
        while d * d <= phi {
            if phi % d == 0 {
                if Self::mod_pow(g, phi / d, q) == 1 {
                    return false;
                }
                if Self::mod_pow(g, d, q) == 1 {
                    return false;
                }
            }
            d += 1;
        }
        true
    }

    fn mod_pow(base: u32, mut exp: u32, modulus: u32) -> u32 {
        let mut result = 1u64;
        let base64 = base as u64;
        let mod64 = modulus as u64;

        let mut b = base64;
        while exp > 0 {
            if exp & 1 == 1 {
                result = (result * b) % mod64;
            }
            b = (b * b) % mod64;
            exp >>= 1;
        }
        result as u32
    }

    fn mod_inverse(a: u32, q: u32) -> u32 {
        Self::mod_pow(a, q - 2, q)
    }

    fn mul_mod(a: u32, b: u32, q: u32) -> u32 {
        ((a as u64 * b as u64) % q as u64) as u32
    }

    pub fn n(&self) -> usize { self.n }
    pub fn q(&self) -> u32 { self.q }
}

/// 32-bit NTT with NEON acceleration
pub struct Ntt32 {
    tables: Ntt32Tables,
}

impl Ntt32 {
    pub fn new(tables: Ntt32Tables) -> Self {
        Self { tables }
    }

    /// Forward NTT using 4-lane NEON butterflies
    #[cfg(target_arch = "aarch64")]
    pub fn forward(&self, coeffs: &mut [u32]) {
        let n = self.tables.n;
        let q = self.tables.q;
        let q_inv_neg = self.tables.q_inv_neg;
        let r2 = self.tables.r2_mod_q;
        let twiddles = &self.tables.twiddles_mont;

        // Convert to Montgomery form and pre-twist
        for i in 0..n {
            let a_mont = Ntt32Tables::to_montgomery(coeffs[i], q, r2, q_inv_neg);
            coeffs[i] = self.montgomery_mul(a_mont, twiddles[i]);
        }

        // Bit-reversal
        self.bit_reverse(coeffs);

        // Cooley-Tukey with 4-lane NEON butterflies
        let log_n = self.tables.log_n;
        for s in 0..log_n {
            let m = 1 << (s + 1);
            let half_m = m / 2;
            let step = n / m;

            for k in (0..n).step_by(m) {
                // Process 4 butterflies at a time when possible
                let mut j = 0;
                while j + 4 <= half_m {
                    let tw_idx0 = 2 * j * step;
                    let tw_idx1 = 2 * (j + 1) * step;
                    let tw_idx2 = 2 * (j + 2) * step;
                    let tw_idx3 = 2 * (j + 3) * step;

                    unsafe {
                        let a_arr = [
                            coeffs[k + j],
                            coeffs[k + j + 1],
                            coeffs[k + j + 2],
                            coeffs[k + j + 3],
                        ];
                        let b_arr = [
                            coeffs[k + j + half_m],
                            coeffs[k + j + 1 + half_m],
                            coeffs[k + j + 2 + half_m],
                            coeffs[k + j + 3 + half_m],
                        ];
                        let w_arr = [
                            self.get_twiddle(tw_idx0, twiddles),
                            self.get_twiddle(tw_idx1, twiddles),
                            self.get_twiddle(tw_idx2, twiddles),
                            self.get_twiddle(tw_idx3, twiddles),
                        ];

                        let a = vld1q_u32(a_arr.as_ptr());
                        let b = vld1q_u32(b_arr.as_ptr());
                        let w = vld1q_u32(w_arr.as_ptr());

                        let (a_out, b_out) = butterfly_ct_neon_4x32(a, b, w, q, q_inv_neg);

                        let a_result: [u32; 4] = std::mem::transmute(a_out);
                        let b_result: [u32; 4] = std::mem::transmute(b_out);

                        coeffs[k + j] = a_result[0];
                        coeffs[k + j + 1] = a_result[1];
                        coeffs[k + j + 2] = a_result[2];
                        coeffs[k + j + 3] = a_result[3];
                        coeffs[k + j + half_m] = b_result[0];
                        coeffs[k + j + 1 + half_m] = b_result[1];
                        coeffs[k + j + 2 + half_m] = b_result[2];
                        coeffs[k + j + 3 + half_m] = b_result[3];
                    }
                    j += 4;
                }

                // Handle remaining butterflies (< 4)
                while j < half_m {
                    let tw_idx = 2 * j * step;
                    let w = self.get_twiddle(tw_idx, twiddles);
                    let (a_out, b_out) = butterfly_ct_scalar_32(
                        coeffs[k + j],
                        coeffs[k + j + half_m],
                        w,
                        q,
                        q_inv_neg,
                    );
                    coeffs[k + j] = a_out;
                    coeffs[k + j + half_m] = b_out;
                    j += 1;
                }
            }
        }

        // Convert from Montgomery form
        for c in coeffs.iter_mut() {
            *c = self.from_montgomery(*c);
        }
    }

    /// Inverse NTT using 4-lane NEON butterflies
    #[cfg(target_arch = "aarch64")]
    pub fn inverse(&self, coeffs: &mut [u32]) {
        let n = self.tables.n;
        let q = self.tables.q;
        let q_inv_neg = self.tables.q_inv_neg;
        let r2 = self.tables.r2_mod_q;
        let inv_twiddles = &self.tables.inv_twiddles_mont;

        // Convert to Montgomery form
        for c in coeffs.iter_mut() {
            *c = Ntt32Tables::to_montgomery(*c, q, r2, q_inv_neg);
        }

        // Gentleman-Sande with 4-lane NEON butterflies
        let log_n = self.tables.log_n;
        for s in (0..log_n).rev() {
            let m = 1 << (s + 1);
            let half_m = m / 2;
            let step = n / m;

            for k in (0..n).step_by(m) {
                let mut j = 0;
                while j + 4 <= half_m {
                    let tw_idx0 = 2 * j * step;
                    let tw_idx1 = 2 * (j + 1) * step;
                    let tw_idx2 = 2 * (j + 2) * step;
                    let tw_idx3 = 2 * (j + 3) * step;

                    unsafe {
                        let a_arr = [
                            coeffs[k + j],
                            coeffs[k + j + 1],
                            coeffs[k + j + 2],
                            coeffs[k + j + 3],
                        ];
                        let b_arr = [
                            coeffs[k + j + half_m],
                            coeffs[k + j + 1 + half_m],
                            coeffs[k + j + 2 + half_m],
                            coeffs[k + j + 3 + half_m],
                        ];
                        let w_arr = [
                            self.get_inv_twiddle(tw_idx0, inv_twiddles),
                            self.get_inv_twiddle(tw_idx1, inv_twiddles),
                            self.get_inv_twiddle(tw_idx2, inv_twiddles),
                            self.get_inv_twiddle(tw_idx3, inv_twiddles),
                        ];

                        let a = vld1q_u32(a_arr.as_ptr());
                        let b = vld1q_u32(b_arr.as_ptr());
                        let w = vld1q_u32(w_arr.as_ptr());

                        let (a_out, b_out) = butterfly_gs_neon_4x32(a, b, w, q, q_inv_neg);

                        let a_result: [u32; 4] = std::mem::transmute(a_out);
                        let b_result: [u32; 4] = std::mem::transmute(b_out);

                        coeffs[k + j] = a_result[0];
                        coeffs[k + j + 1] = a_result[1];
                        coeffs[k + j + 2] = a_result[2];
                        coeffs[k + j + 3] = a_result[3];
                        coeffs[k + j + half_m] = b_result[0];
                        coeffs[k + j + 1 + half_m] = b_result[1];
                        coeffs[k + j + 2 + half_m] = b_result[2];
                        coeffs[k + j + 3 + half_m] = b_result[3];
                    }
                    j += 4;
                }

                while j < half_m {
                    let tw_idx = 2 * j * step;
                    let w = self.get_inv_twiddle(tw_idx, inv_twiddles);
                    let (a_out, b_out) = butterfly_gs_scalar_32(
                        coeffs[k + j],
                        coeffs[k + j + half_m],
                        w,
                        q,
                        q_inv_neg,
                    );
                    coeffs[k + j] = a_out;
                    coeffs[k + j + half_m] = b_out;
                    j += 1;
                }
            }
        }

        // Bit-reversal
        self.bit_reverse(coeffs);

        // Scale by n^(-1) and post-untwist, then convert from Montgomery
        let n_inv_mont = self.tables.n_inv_mont;
        for i in 0..n {
            let scaled = self.montgomery_mul(coeffs[i], n_inv_mont);
            let untwisted = self.montgomery_mul(scaled, inv_twiddles[i]);
            coeffs[i] = self.from_montgomery(untwisted);
        }
    }

    /// Scalar fallback for non-aarch64
    #[cfg(not(target_arch = "aarch64"))]
    pub fn forward(&self, coeffs: &mut [u32]) {
        self.forward_scalar(coeffs);
    }

    #[cfg(not(target_arch = "aarch64"))]
    pub fn inverse(&self, coeffs: &mut [u32]) {
        self.inverse_scalar(coeffs);
    }

    /// Scalar forward NTT (reference / fallback)
    pub fn forward_scalar(&self, coeffs: &mut [u32]) {
        let n = self.tables.n;
        let q = self.tables.q;
        let q_inv_neg = self.tables.q_inv_neg;
        let r2 = self.tables.r2_mod_q;
        let twiddles = &self.tables.twiddles_mont;

        // Convert to Montgomery form and pre-twist
        for i in 0..n {
            let a_mont = Ntt32Tables::to_montgomery(coeffs[i], q, r2, q_inv_neg);
            coeffs[i] = self.montgomery_mul(a_mont, twiddles[i]);
        }

        self.bit_reverse(coeffs);

        let log_n = self.tables.log_n;
        for s in 0..log_n {
            let m = 1 << (s + 1);
            let half_m = m / 2;
            let step = n / m;

            for k in (0..n).step_by(m) {
                for j in 0..half_m {
                    let tw_idx = 2 * j * step;
                    let w = self.get_twiddle(tw_idx, twiddles);

                    #[cfg(target_arch = "aarch64")]
                    let (a_out, b_out) = butterfly_ct_scalar_32(
                        coeffs[k + j], coeffs[k + j + half_m], w, q, q_inv_neg
                    );

                    #[cfg(not(target_arch = "aarch64"))]
                    let (a_out, b_out) = {
                        let t = self.montgomery_mul(coeffs[k + j + half_m], w);
                        let a = coeffs[k + j];
                        let a_out = if a + t >= q { a + t - q } else { a + t };
                        let b_out = if a >= t { a - t } else { a + q - t };
                        (a_out, b_out)
                    };

                    coeffs[k + j] = a_out;
                    coeffs[k + j + half_m] = b_out;
                }
            }
        }

        for c in coeffs.iter_mut() {
            *c = self.from_montgomery(*c);
        }
    }

    /// Scalar inverse NTT (reference / fallback)
    pub fn inverse_scalar(&self, coeffs: &mut [u32]) {
        let n = self.tables.n;
        let q = self.tables.q;
        let q_inv_neg = self.tables.q_inv_neg;
        let r2 = self.tables.r2_mod_q;
        let inv_twiddles = &self.tables.inv_twiddles_mont;

        for c in coeffs.iter_mut() {
            *c = Ntt32Tables::to_montgomery(*c, q, r2, q_inv_neg);
        }

        let log_n = self.tables.log_n;
        for s in (0..log_n).rev() {
            let m = 1 << (s + 1);
            let half_m = m / 2;
            let step = n / m;

            for k in (0..n).step_by(m) {
                for j in 0..half_m {
                    let tw_idx = 2 * j * step;
                    let w = self.get_inv_twiddle(tw_idx, inv_twiddles);

                    #[cfg(target_arch = "aarch64")]
                    let (a_out, b_out) = butterfly_gs_scalar_32(
                        coeffs[k + j], coeffs[k + j + half_m], w, q, q_inv_neg
                    );

                    #[cfg(not(target_arch = "aarch64"))]
                    let (a_out, b_out) = {
                        let a = coeffs[k + j];
                        let b = coeffs[k + j + half_m];
                        let a_out = if a + b >= q { a + b - q } else { a + b };
                        let diff = if a >= b { a - b } else { a + q - b };
                        let b_out = self.montgomery_mul(diff, w);
                        (a_out, b_out)
                    };

                    coeffs[k + j] = a_out;
                    coeffs[k + j + half_m] = b_out;
                }
            }
        }

        self.bit_reverse(coeffs);

        let n_inv_mont = self.tables.n_inv_mont;
        for i in 0..n {
            let scaled = self.montgomery_mul(coeffs[i], n_inv_mont);
            let untwisted = self.montgomery_mul(scaled, inv_twiddles[i]);
            coeffs[i] = self.from_montgomery(untwisted);
        }
    }

    #[inline]
    fn bit_reverse(&self, coeffs: &mut [u32]) {
        let n = coeffs.len();
        for i in 0..n {
            let j = self.tables.bit_rev_table[i];
            if i < j {
                coeffs.swap(i, j);
            }
        }
    }

    #[inline]
    fn get_twiddle(&self, idx: usize, twiddles: &[u32]) -> u32 {
        let n = self.tables.n;
        if idx == 0 {
            // 1 in Montgomery form
            Ntt32Tables::to_montgomery(1, self.tables.q, self.tables.r2_mod_q, self.tables.q_inv_neg)
        } else if idx < n {
            twiddles[idx]
        } else {
            // -psi^k = q - psi^k
            self.tables.q - twiddles[idx - n]
        }
    }

    #[inline]
    fn get_inv_twiddle(&self, idx: usize, inv_twiddles: &[u32]) -> u32 {
        let n = self.tables.n;
        if idx == 0 {
            Ntt32Tables::to_montgomery(1, self.tables.q, self.tables.r2_mod_q, self.tables.q_inv_neg)
        } else if idx < n {
            inv_twiddles[idx]
        } else {
            self.tables.q - inv_twiddles[idx - n]
        }
    }

    #[inline]
    fn montgomery_mul(&self, a: u32, b: u32) -> u32 {
        let prod = a as u64 * b as u64;
        let t_lo = prod as u32;
        let m = t_lo.wrapping_mul(self.tables.q_inv_neg);
        let mq = m as u64 * self.tables.q as u64;
        let sum = prod.wrapping_add(mq);
        let result = (sum >> 32) as u32;
        if result >= self.tables.q { result - self.tables.q } else { result }
    }

    #[inline]
    fn from_montgomery(&self, a: u32) -> u32 {
        let m = a.wrapping_mul(self.tables.q_inv_neg);
        let mq = m as u64 * self.tables.q as u64;
        let sum = (a as u64).wrapping_add(mq);
        let result = (sum >> 32) as u32;
        if result >= self.tables.q { result - self.tables.q } else { result }
    }
}

#[cfg(test)]
mod tests {
    use crate::ntt32::*;
use crate::butterfly::*;

    // 30-bit NTT-friendly prime: q ≡ 1 (mod 2*4096)
    const TEST_Q: u32 = 1073479681;

    #[test]
    fn test_ntt32_roundtrip() {
        let tables = Ntt32Tables::new(1024, TEST_Q);
        let ntt = Ntt32::new(tables);

        let mut coeffs: Vec<u32> = (0..1024).map(|i| (i * 7) as u32 % TEST_Q).collect();
        let original = coeffs.clone();

        ntt.forward(&mut coeffs);
        ntt.inverse(&mut coeffs);

        assert_eq!(coeffs, original, "NTT roundtrip failed");
    }

    #[test]
    fn test_ntt32_scalar_roundtrip() {
        let tables = Ntt32Tables::new(1024, TEST_Q);
        let ntt = Ntt32::new(tables);

        let mut coeffs: Vec<u32> = (0..1024).map(|i| (i * 13) as u32 % TEST_Q).collect();
        let original = coeffs.clone();

        ntt.forward_scalar(&mut coeffs);
        ntt.inverse_scalar(&mut coeffs);

        assert_eq!(coeffs, original, "Scalar NTT roundtrip failed");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_ntt32_neon_matches_scalar() {
        let tables = Ntt32Tables::new(1024, TEST_Q);
        let ntt = Ntt32::new(tables);

        let input: Vec<u32> = (0..1024).map(|i| (i * 17) as u32 % TEST_Q).collect();

        let mut neon_coeffs = input.clone();
        let mut scalar_coeffs = input.clone();

        ntt.forward(&mut neon_coeffs);
        ntt.forward_scalar(&mut scalar_coeffs);

        assert_eq!(neon_coeffs, scalar_coeffs, "NEON forward NTT doesn't match scalar");

        ntt.inverse(&mut neon_coeffs);
        ntt.inverse_scalar(&mut scalar_coeffs);

        assert_eq!(neon_coeffs, scalar_coeffs, "NEON inverse NTT doesn't match scalar");
    }
}
