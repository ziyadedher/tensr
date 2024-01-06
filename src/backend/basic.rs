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
    fn test_scalar_zero() {
        let zero: u8 = Backend::t0_zero();
        assert_eq!(zero, 0);
    }

    #[test]
    fn test_scalar_one() {
        let one: u8 = Backend::t0_one();
        assert_eq!(one, 1);
    }

    #[test]
    fn test_vector_zeros() {
        let zeros: Vec<u8> = Backend::t1_zeros(3);
        assert_eq!(zeros, vec![0, 0, 0]);
    }

    #[test]
    fn test_vector_ones() {
        let ones: Vec<u8> = Backend::t1_ones(3);
        assert_eq!(ones, vec![1, 1, 1]);
    }

    #[test]
    fn test_matrix_zeros() {
        let zeros: Vec<Vec<u8>> = Backend::t2_zeros(3, 3);
        assert_eq!(zeros, vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]]);
    }

    #[test]
    fn test_matrix_ones() {
        let ones: Vec<Vec<u8>> = Backend::t2_ones(3, 3);
        assert_eq!(ones, vec![vec![1, 1, 1], vec![1, 1, 1], vec![1, 1, 1]]);
    }

    #[test]
    fn test_matrix_identity() {
        let eye: Vec<Vec<u8>> = Backend::t2_identity(3);
        assert_eq!(eye, vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]);
    }

    #[test]
    fn test_scalar_scalar_add() {
        assert_eq!(Backend::t0_t0_add(1, 2), 3);
    }

    #[test]
    fn test_vector_scalar_add() {
        assert_eq!(Backend::t1_t0_add(vec![1, 2, 3], 2), vec![3, 4, 5]);
    }

    #[test]
    fn test_vector_vector_add() {
        assert_eq!(
            Backend::t1_t1_add(vec![1, 2, 3], vec![4, 5, 6]),
            vec![5, 7, 9]
        );
    }

    #[test]
    fn test_matrix_scalar_add() {
        assert_eq!(
            Backend::t2_t0_add(vec![vec![1, 2, 3], vec![4, 5, 6]], 2),
            vec![vec![3, 4, 5], vec![6, 7, 8]]
        );
    }

    #[test]
    fn test_matrix_vector_add_along_0() {
        assert_eq!(
            Backend::t2_t1_add(vec![vec![1, 2, 3], vec![4, 5, 6]], vec![2, 3], 0),
            vec![vec![3, 4, 5], vec![7, 8, 9]]
        );
    }

    #[test]
    fn test_matrix_vector_add_along_1() {
        assert_eq!(
            Backend::t2_t1_add(vec![vec![1, 2, 3], vec![4, 5, 6]], vec![2, 3, 4], 1),
            vec![vec![3, 5, 7], vec![6, 8, 10]]
        );
    }

    #[test]
    fn test_matrix_matrix_add() {
        assert_eq!(
            Backend::t2_t2_add(
                vec![vec![1, 2, 3], vec![4, 5, 6]],
                vec![vec![2, 3, 4], vec![5, 6, 7]]
            ),
            vec![vec![3, 5, 7], vec![9, 11, 13]]
        );
    }

    #[test]
    fn test_dot() {
        assert_eq!(Backend::t1_t1_dot(vec![1, 2, 3], vec![4, 5, 6]), 32);
    }

    #[test]
    fn test_matmul() {
        assert_eq!(
            Backend::t2_t2_matmul(
                vec![vec![1, 2, 3], vec![4, 5, 6]],
                vec![vec![2, 3], vec![4, 5], vec![6, 7]]
            ),
            vec![vec![28, 34], vec![64, 79]]
        );
    }
}
