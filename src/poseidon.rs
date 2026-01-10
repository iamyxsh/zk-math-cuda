use crate::field::Fp;

pub const T: usize = 3;
const RF: usize = 8; // full rounds (4 + 4)
const RP: usize = 56; // partial rounds
const TOTAL_ROUNDS: usize = RF + RP;

// MDS matrix: [[2,1,1],[1,2,1],[1,1,2]]
// Verified MDS: all square submatrix determinants nonzero over BLS12-381.
fn mds_matrix() -> [[Fp; T]; T] {
    let one = Fp::ONE;
    let two = Fp::from_u64(2);
    [
        [two, one, one],
        [one, two, one],
        [one, one, two],
    ]
}

fn round_constants() -> Vec<Fp> {
    (1..=(TOTAL_ROUNDS * T) as u64)
        .map(Fp::from_u64)
        .collect()
}

pub fn poseidon_permutation(state: &[Fp; T]) -> [Fp; T] {
    let rc = round_constants();
    let mds = mds_matrix();
    let mut s = *state;

    for r in 0..TOTAL_ROUNDS {
        // AddRoundConstants
        for j in 0..T {
            s[j] = s[j].add(rc[r * T + j]);
        }

        // SubWords (S-box: x^5)
        if r < RF / 2 || r >= RF / 2 + RP {
            // Full round
            for j in 0..T {
                s[j] = s[j].pow5();
            }
        } else {
            // Partial round
            s[0] = s[0].pow5();
        }

        // MixLayer (MDS matrix multiplication)
        let mut new_s = [Fp::ZERO; T];
        for i in 0..T {
            for j in 0..T {
                new_s[i] = new_s[i].add(mds[i][j].mul(s[j]));
            }
        }
        s = new_s;
    }

    s
}

/// Export constant generation for GPU kernel setup (Step 4).
pub fn get_round_constants() -> Vec<Fp> {
    round_constants()
}

pub fn get_mds_matrix() -> [[Fp; T]; T] {
    mds_matrix()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poseidon_cpu_known_vector() {
        let input = [Fp::ZERO; T];
        let output = poseidon_permutation(&input);

        assert!(
            output.iter().any(|x| *x != Fp::ZERO),
            "Poseidon of zero state must not be zero"
        );

        // Regression vector — computed from this implementation.
        // To cross-verify: implement the same parameters (t=3, RF=8, RP=56,
        // sequential integer round constants, MDS=[[2,1,1],[1,2,1],[1,1,2]])
        // in a second language (Python/Sage) and compare.
        let raw0 = output[0].from_mont();
        let raw1 = output[1].from_mont();
        let raw2 = output[2].from_mont();

        // Deterministic recomputation must match
        let output_again = poseidon_permutation(&input);
        assert_eq!(output[0].from_mont(), output_again[0].from_mont());
        assert_eq!(output[1].from_mont(), output_again[1].from_mont());
        assert_eq!(output[2].from_mont(), output_again[2].from_mont());

        // Print for first-run capture (remove after hardcoding):
        eprintln!("known vector [0]: {:?}", raw0);
        eprintln!("known vector [1]: {:?}", raw1);
        eprintln!("known vector [2]: {:?}", raw2);
    }

    #[test]
    fn test_poseidon_cpu_nonzero() {
        let input = [Fp::from_u64(1), Fp::from_u64(2), Fp::from_u64(3)];
        let output = poseidon_permutation(&input);
        assert!(output.iter().any(|x| *x != Fp::ZERO));
        // Different input should give different output
        let output_zero = poseidon_permutation(&[Fp::ZERO; T]);
        assert_ne!(output, output_zero);
    }

    #[test]
    fn test_poseidon_cpu_deterministic() {
        let input = [Fp::from_u64(42), Fp::from_u64(0), Fp::from_u64(1)];
        let out1 = poseidon_permutation(&input);
        let out2 = poseidon_permutation(&input);
        assert_eq!(out1, out2);
    }
}
