/// BLS12-381 scalar field element in Montgomery form.
/// p = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp(pub [u64; 4]);

const MODULUS: [u64; 4] = [
    0xffff_ffff_0000_0001,
    0x53bd_a402_fffe_5bfe,
    0x3339_d808_09a1_d805,
    0x73ed_a753_299d_7d48,
];

// -p^(-1) mod 2^64
// Derivation: p[0] = 1 - 2^32, so p[0]^(-1) = 1 + 2^32,
// and -p[0]^(-1) = -(1 + 2^32) = 2^64 - 1 - 2^32
const INV: u64 = 0xffff_fffe_ffff_ffff;

// R = 2^256 mod p (this is "1" in Montgomery form)
const R: [u64; 4] = [
    0x0000_0001_ffff_fffe,
    0x5884_b7fa_0003_4802,
    0x998c_4fef_ecbc_4ff5,
    0x1824_b159_acc5_056f,
];

// R^2 mod p (used to convert into Montgomery form)
const R2: [u64; 4] = [
    0xc999_e990_f3f2_9c6d,
    0x2b6c_edcb_8792_5c23,
    0x05d3_1496_7254_398f,
    0x0748_d9d9_9f59_ff11,
];

impl Fp {
    pub const ZERO: Self = Fp([0; 4]);
    pub const ONE: Self = Fp(R);

    pub fn from_u64(val: u64) -> Self {
        // val * R^2 * R^(-1) = val * R = val in Montgomery form
        Fp([val, 0, 0, 0]).mont_mul(&Fp(R2))
    }

    pub fn from_mont(self) -> [u64; 4] {
        // a_mont * 1 * R^(-1) = a
        self.mont_mul(&Fp([1, 0, 0, 0])).0
    }

    pub fn add(self, other: Self) -> Self {
        let mut r = [0u64; 4];
        let mut carry = 0u128;
        for i in 0..4 {
            let s = (self.0[i] as u128) + (other.0[i] as u128) + carry;
            r[i] = s as u64;
            carry = s >> 64;
        }
        if carry > 0 || gte(&r, &MODULUS) {
            sub_assign(&mut r, &MODULUS);
        }
        Fp(r)
    }

    pub fn sub(self, other: Self) -> Self {
        let mut r = [0u64; 4];
        let mut borrow = 0u64;
        for i in 0..4 {
            let (d1, b1) = self.0[i].overflowing_sub(other.0[i]);
            let (d2, b2) = d1.overflowing_sub(borrow);
            r[i] = d2;
            borrow = (b1 as u64) + (b2 as u64);
        }
        if borrow > 0 {
            let mut carry = 0u128;
            for i in 0..4 {
                let s = (r[i] as u128) + (MODULUS[i] as u128) + carry;
                r[i] = s as u64;
                carry = s >> 64;
            }
        }
        Fp(r)
    }

    pub fn mul(self, other: Self) -> Self {
        self.mont_mul(&other)
    }

    pub fn pow5(self) -> Self {
        let x2 = self.mul(self);
        let x4 = x2.mul(x2);
        x4.mul(self)
    }

    // CIOS Montgomery multiplication: a * b * R^(-1) mod p
    fn mont_mul(&self, other: &Fp) -> Fp {
        let a = &self.0;
        let b = &other.0;
        let mut t = [0u64; 5];

        for i in 0..4 {
            let mut c: u64 = 0;
            for j in 0..4 {
                let x = (t[j] as u128) + (a[j] as u128) * (b[i] as u128) + (c as u128);
                t[j] = x as u64;
                c = (x >> 64) as u64;
            }
            let x = (t[4] as u128) + (c as u128);
            t[4] = x as u64;

            let m = t[0].wrapping_mul(INV);
            let x = (t[0] as u128) + (m as u128) * (MODULUS[0] as u128);
            c = (x >> 64) as u64;

            for j in 1..4 {
                let x = (t[j] as u128) + (m as u128) * (MODULUS[j] as u128) + (c as u128);
                t[j - 1] = x as u64;
                c = (x >> 64) as u64;
            }
            let x = (t[4] as u128) + (c as u128);
            t[3] = x as u64;
            t[4] = (x >> 64) as u64;
        }

        let mut result = [t[0], t[1], t[2], t[3]];
        if t[4] > 0 || gte(&result, &MODULUS) {
            sub_assign(&mut result, &MODULUS);
        }
        Fp(result)
    }
}

fn gte(a: &[u64; 4], b: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
        if a[i] > b[i] { return true; }
        if a[i] < b[i] { return false; }
    }
    true
}

fn sub_assign(a: &mut [u64; 4], b: &[u64; 4]) {
    let mut borrow = 0u64;
    for i in 0..4 {
        let (d1, b1) = a[i].overflowing_sub(b[i]);
        let (d2, b2) = d1.overflowing_sub(borrow);
        a[i] = d2;
        borrow = (b1 as u64) + (b2 as u64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_mul_identity() {
        let a = Fp::from_u64(42);
        assert_eq!(a.mul(Fp::ONE), a);
    }

    #[test]
    fn test_field_mul_commutative() {
        let a = Fp::from_u64(7);
        let b = Fp::from_u64(13);
        assert_eq!(a.mul(b), b.mul(a));
    }

    #[test]
    fn test_field_add_sub_inverse() {
        let a = Fp::from_u64(100);
        let b = Fp::from_u64(42);
        assert_eq!(a.add(b).sub(b), a);
    }

    #[test]
    fn test_field_mul_zero() {
        let a = Fp::from_u64(12345);
        assert_eq!(a.mul(Fp::ZERO), Fp::ZERO);
    }

    #[test]
    fn test_field_from_u64_roundtrip() {
        let val = 9999u64;
        let fp = Fp::from_u64(val);
        let raw = fp.from_mont();
        assert_eq!(raw, [val, 0, 0, 0]);
    }

    #[test]
    fn test_field_pow5() {
        let two = Fp::from_u64(2);
        let result = two.pow5();
        assert_eq!(result, Fp::from_u64(32));
    }

    #[test]
    fn test_field_add_overflow() {
        // p - 1 + 1 should wrap to 0
        let p_minus_1 = Fp::ZERO.sub(Fp::ONE); // -1 mod p = p - 1
        let result = p_minus_1.add(Fp::ONE);
        assert_eq!(result, Fp::ZERO);
    }

    #[test]
    fn test_field_sub_underflow() {
        // 0 - 1 = p - 1
        let result = Fp::ZERO.sub(Fp::ONE);
        let p_minus_1 = result.from_mont();
        // p - 1 in normal form: MODULUS - 1
        assert_eq!(p_minus_1[0], MODULUS[0] - 1);
        assert_eq!(p_minus_1[1], MODULUS[1]);
        assert_eq!(p_minus_1[2], MODULUS[2]);
        assert_eq!(p_minus_1[3], MODULUS[3]);
    }

    #[test]
    fn test_field_mul_associative() {
        let a = Fp::from_u64(3);
        let b = Fp::from_u64(5);
        let c = Fp::from_u64(7);
        assert_eq!(a.mul(b).mul(c), a.mul(b.mul(c)));
    }

    #[test]
    fn test_field_distributive() {
        let a = Fp::from_u64(3);
        let b = Fp::from_u64(5);
        let c = Fp::from_u64(7);
        // a * (b + c) = a*b + a*c
        assert_eq!(a.mul(b.add(c)), a.mul(b).add(a.mul(c)));
    }
}
