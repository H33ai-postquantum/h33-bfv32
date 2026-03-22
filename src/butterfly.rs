//! 32-bit NEON butterfly operations for NTT
//! Open source — MIT License
//! Part of H33 BFV32 library

pub fn compute_q_inv_neg32(q: u32) -> u32 {
    let mut x = q; // q is odd for NTT primes
    for _ in 0..5 {
        x = x.wrapping_mul(2u32.wrapping_sub(q.wrapping_mul(x)));
    }
    x.wrapping_neg()
}

pub fn compute_r2_mod_q32(q: u32) -> u32 {
    // R mod q = 2^32 mod q
    let mut r = 1u32;
    for _ in 0..32 {
        r = ((r as u64) << 1).rem_euclid(q as u64) as u32;
    }
    // R^2 mod q
    ((r as u64 * r as u64) % q as u64) as u32
}

pub fn butterfly_ct_scalar_32(a: u32, b: u32, w_mont: u32, q: u32, q_inv_neg: u32) -> (u32, u32) {
    // t = montgomery_reduce(b * w_mont)
    let prod = b as u64 * w_mont as u64;
    let t_lo = prod as u32;
    let m = t_lo.wrapping_mul(q_inv_neg);
    let mq = m as u64 * q as u64;
    let sum = prod.wrapping_add(mq);
    let t = (sum >> 32) as u32;
    let t = if t >= q { t - q } else { t };

    // a_out = (a + t) mod q
    let a_out = {
        let s = a + t;
        if s >= q { s - q } else { s }
    };

    // b_out = (a - t) mod q
    let b_out = if a >= t { a - t } else { a + q - t };

    (a_out, b_out)
}

pub fn butterfly_gs_scalar_32(a: u32, b: u32, w_mont: u32, q: u32, q_inv_neg: u32) -> (u32, u32) {
    // a_out = (a + b) mod q
    let a_out = {
        let s = a + b;
        if s >= q { s - q } else { s }
    };

    // diff = (a - b) mod q
    let diff = if a >= b { a - b } else { a + q - b };

    // b_out = montgomery_reduce(diff * w_mont)
    let prod = diff as u64 * w_mont as u64;
    let t_lo = prod as u32;
    let m = t_lo.wrapping_mul(q_inv_neg);
    let mq = m as u64 * q as u64;
    let sum = prod.wrapping_add(mq);
    let t = (sum >> 32) as u32;
    let b_out = if t >= q { t - q } else { t };

    (a_out, b_out)
}

pub fn butterfly_ct_neon_2x(
    a: [u64; 2],
    b: [u64; 2],
    w: [u64; 2],
    q: u64,
) -> ([u64; 2], [u64; 2]) {
    use super::butterfly::butterfly_ct_scalar;
    let (a0, b0) = butterfly_ct_scalar(a[0], b[0], w[0], q);
    let (a1, b1) = butterfly_ct_scalar(a[1], b[1], w[1], q);
    ([a0, a1], [b0, b1])
}

pub fn butterfly_gs_neon_2x(
    a: [u64; 2],
    b: [u64; 2],
    w: [u64; 2],
    q: u64,
) -> ([u64; 2], [u64; 2]) {
    use super::butterfly::butterfly_gs_scalar;
    let (a0, b0) = butterfly_gs_scalar(a[0], b[0], w[0], q);
    let (a1, b1) = butterfly_gs_scalar(a[1], b[1], w[1], q);
    ([a0, a1], [b0, b1])
}
