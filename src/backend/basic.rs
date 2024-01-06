use std::ops::{Add, Mul};

use crate::backend::Backend as BackendTrait;

#[derive(Clone, Debug)]
pub struct Backend {}

impl<T: Clone> BackendTrait<T> for Backend {
    type Index = usize;
    type Dimension = usize;
    type T0Repr = T;
    type T1Repr = Vec<T>;
    type T2Repr = Vec<Vec<T>>;
    type T3Repr = Vec<Vec<Vec<T>>>;

    fn t0_zero() -> Self::T0Repr
    where
        T: From<u8>,
    {
        0.into()
    }

    fn t0_one() -> Self::T0Repr
    where
        T: From<u8>,
    {
        1.into()
    }

    fn t1_zeros(d0: usize) -> Self::T1Repr
    where
        T: From<u8> + Copy,
    {
        vec![0.into(); d0]
    }

    fn t1_ones(d0: usize) -> Self::T1Repr
    where
        T: From<u8> + Copy,
    {
        vec![1.into(); d0]
    }

    fn t2_zeros(d0: usize, d1: usize) -> Self::T2Repr
    where
        T: From<u8> + Copy,
    {
        vec![vec![0.into(); d1]; d0]
    }

    fn t2_ones(d0: usize, d1: usize) -> Self::T2Repr
    where
        T: From<u8> + Copy,
    {
        vec![vec![1.into(); d1]; d0]
    }

    fn t3_zeros(d0: usize, d1: usize, d2: usize) -> Self::T3Repr
    where
        T: From<u8> + Copy,
    {
        vec![vec![vec![0.into(); d2]; d1]; d0]
    }

    fn t3_ones(d0: Self::Index, d1: Self::Index, d2: Self::Index) -> Self::T3Repr
    where
        T: From<u8> + Copy,
    {
        vec![vec![vec![1.into(); d2]; d1]; d0]
    }

    fn t2_identity(d: usize) -> Self::T2Repr
    where
        T: From<u8> + Copy,
    {
        let mut matrix = Self::t2_zeros(d, d);
        for i in 0..d {
            matrix[i][i] = 1.into();
        }
        matrix
    }

    fn t2_transpose(a: Self::T2Repr) -> Self::T2Repr
    where
        T: From<u8> + Copy,
    {
        let mut result = Self::t2_zeros(a[0].len(), a.len());
        for i in 0..a.len() {
            for j in 0..a[0].len() {
                result[j][i] = a[i][j];
            }
        }
        result
    }

    fn t3_permute(
        a: Self::T3Repr,
        p: (Self::Dimension, Self::Dimension, Self::Dimension),
    ) -> Self::T3Repr
    where
        T: From<u8> + Copy,
    {
        let d = [a.len(), a[0].len(), a[0][0].len()];
        let (pd0, pd1, pd2) = (d[p.0], d[p.1], d[p.2]);
        let mut result = Self::t3_zeros(pd0, pd1, pd2);
        for i in 0..a.len() {
            for j in 0..a[0].len() {
                for k in 0..a[0][0].len() {
                    let idx = [i, j, k];
                    let (pi, pj, pk) = (idx[p.0], idx[p.1], idx[p.2]);
                    result[pi][pj][pk] = a[i][j][k];
                }
            }
        }
        result
    }

    fn t0_t0_add(a: Self::T0Repr, b: Self::T0Repr) -> Self::T0Repr
    where
        T: Add<Output = T>,
    {
        a + b
    }

    fn t1_t0_add(a: Self::T1Repr, b: Self::T0Repr) -> Self::T1Repr
    where
        T: Add<Output = T> + Copy,
    {
        a.into_iter().map(|a| a + b).collect()
    }

    fn t1_t1_add(a: Self::T1Repr, b: Self::T1Repr) -> Self::T1Repr
    where
        T: Add<Output = T>,
    {
        assert_eq!(a.len(), b.len());
        a.into_iter()
            .zip(b.into_iter())
            .map(|(a, b)| a + b)
            .collect()
    }

    fn t2_t0_add(a: Self::T2Repr, b: Self::T0Repr) -> Self::T2Repr
    where
        T: Add<Output = T> + Copy,
    {
        a.into_iter()
            .map(|a| a.into_iter().map(|a| a + b).collect())
            .collect()
    }

    fn t2_t1_add(a: Self::T2Repr, b: Self::T1Repr, along: Self::Dimension) -> Self::T2Repr
    where
        T: Add<Output = T> + Copy,
    {
        match along {
            0 => {
                assert_eq!(a.len(), b.len());
                a.into_iter()
                    .zip(b.into_iter())
                    .map(|(a, b)| a.into_iter().map(|a| a + b).collect())
                    .collect()
            }
            1 => {
                assert_eq!(a[0].len(), b.len());
                a.into_iter()
                    .map(|a| a.into_iter().zip(b.iter()).map(|(a, b)| a + *b).collect())
                    .collect()
            }
            _ => unreachable!(),
        }
    }

    fn t2_t2_add(a: Self::T2Repr, b: Self::T2Repr) -> Self::T2Repr
    where
        T: Add<Output = T>,
    {
        a.into_iter()
            .zip(b.into_iter())
            .map(|(a, b)| {
                a.into_iter()
                    .zip(b.into_iter())
                    .map(|(a, b)| a + b)
                    .collect()
            })
            .collect()
    }

    fn t1_t1_dot(a: Self::T1Repr, b: Self::T1Repr) -> Self::T0Repr
    where
        T: Add<Output = T> + Mul<Output = T> + From<u8>,
    {
        a.into_iter()
            .zip(b.into_iter())
            .fold(0.into(), |acc, (a, b)| acc + a * b)
    }

    fn t2_t2_matmul(a: Self::T2Repr, b: Self::T2Repr) -> Self::T2Repr
    where
        T: Add<Output = T> + Mul<Output = T> + From<u8> + Copy,
    {
        let mut result = Self::t2_zeros(a.len(), b[0].len());
        for i in 0..a.len() {
            for j in 0..b[0].len() {
                for k in 0..b.len() {
                    result[i][j] = result[i][j] + a[i][k] * b[k][j];
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t0_zero() {
        let zero: u8 = Backend::t0_zero();
        assert_eq!(zero, 0);
    }

    #[test]
    fn test_t0_one() {
        let one: u8 = Backend::t0_one();
        assert_eq!(one, 1);
    }

    #[test]
    fn test_t1_zeros() {
        let zeros: Vec<u8> = Backend::t1_zeros(3);
        assert_eq!(zeros, vec![0, 0, 0]);
    }

    #[test]
    fn test_t1_ones() {
        let ones: Vec<u8> = Backend::t1_ones(3);
        assert_eq!(ones, vec![1, 1, 1]);
    }

    #[test]
    fn test_t2_zeros() {
        let zeros: Vec<Vec<u8>> = Backend::t2_zeros(2, 3);
        assert_eq!(zeros, vec![vec![0, 0, 0], vec![0, 0, 0]]);
    }

    #[test]
    fn test_t2_ones() {
        let ones: Vec<Vec<u8>> = Backend::t2_ones(2, 3);
        assert_eq!(ones, vec![vec![1, 1, 1], vec![1, 1, 1]]);
    }

    #[test]
    fn test_t3_zeros() {
        let zeros: Vec<Vec<Vec<u8>>> = Backend::t3_zeros(2, 3, 4);
        assert_eq!(
            zeros,
            vec![
                vec![vec![0, 0, 0, 0], vec![0, 0, 0, 0], vec![0, 0, 0, 0]],
                vec![vec![0, 0, 0, 0], vec![0, 0, 0, 0], vec![0, 0, 0, 0]]
            ]
        );
    }

    #[test]
    fn test_t3_ones() {
        let ones: Vec<Vec<Vec<u8>>> = Backend::t3_ones(2, 3, 4);
        assert_eq!(
            ones,
            vec![
                vec![vec![1, 1, 1, 1], vec![1, 1, 1, 1], vec![1, 1, 1, 1]],
                vec![vec![1, 1, 1, 1], vec![1, 1, 1, 1], vec![1, 1, 1, 1]]
            ]
        );
    }

    #[test]
    fn test_t2_identity() {
        let eye: Vec<Vec<u8>> = Backend::t2_identity(3);
        assert_eq!(eye, vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]);
    }

    #[test]
    fn test_t2_transpose() {
        assert_eq!(
            Backend::t2_transpose(vec![vec![1, 2, 3], vec![4, 5, 6]]),
            vec![vec![1, 4], vec![2, 5], vec![3, 6]]
        );
    }

    #[test]
    fn test_t3_permute() {
        let input = vec![
            vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]],
            vec![vec![10, 11, 12], vec![13, 14, 15], vec![16, 17, 18]],
        ];
        let permuted = Backend::t3_permute(input, (2, 0, 1));
        let expected = vec![
            vec![vec![1, 4, 7], vec![10, 13, 16]],
            vec![vec![2, 5, 8], vec![11, 14, 17]],
            vec![vec![3, 6, 9], vec![12, 15, 18]],
        ];
        assert_eq!(permuted, expected);
    }

    #[test]
    fn test_t0_t0_add() {
        assert_eq!(Backend::t0_t0_add(1, 2), 3);
    }

    #[test]
    fn test_t1_t0_add() {
        assert_eq!(Backend::t1_t0_add(vec![1, 2, 3], 2), vec![3, 4, 5]);
    }

    #[test]
    fn test_t1_t1_add() {
        assert_eq!(
            Backend::t1_t1_add(vec![1, 2, 3], vec![4, 5, 6]),
            vec![5, 7, 9]
        );
    }

    #[test]
    fn test_t2_t0_add() {
        assert_eq!(
            Backend::t2_t0_add(vec![vec![1, 2, 3], vec![4, 5, 6]], 2),
            vec![vec![3, 4, 5], vec![6, 7, 8]]
        );
    }

    #[test]
    fn test_t2_t1_add_along_0() {
        assert_eq!(
            Backend::t2_t1_add(vec![vec![1, 2, 3], vec![4, 5, 6]], vec![2, 3], 0),
            vec![vec![3, 4, 5], vec![7, 8, 9]]
        );
    }

    #[test]
    fn test_t2_t1_add_along_1() {
        assert_eq!(
            Backend::t2_t1_add(vec![vec![1, 2, 3], vec![4, 5, 6]], vec![2, 3, 4], 1),
            vec![vec![3, 5, 7], vec![6, 8, 10]]
        );
    }

    #[test]
    fn test_t2_t2_add() {
        assert_eq!(
            Backend::t2_t2_add(
                vec![vec![1, 2, 3], vec![4, 5, 6]],
                vec![vec![2, 3, 4], vec![5, 6, 7]]
            ),
            vec![vec![3, 5, 7], vec![9, 11, 13]]
        );
    }

    #[test]
    fn test_t1_t1_dot() {
        assert_eq!(Backend::t1_t1_dot(vec![1, 2, 3], vec![4, 5, 6]), 32);
    }

    #[test]
    fn test_t2_t2_matmul() {
        assert_eq!(
            Backend::t2_t2_matmul(
                vec![vec![1, 2, 3], vec![4, 5, 6]],
                vec![vec![2, 3], vec![4, 5], vec![6, 7]]
            ),
            vec![vec![28, 34], vec![64, 79]]
        );
    }
}
