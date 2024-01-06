use std::ops::{Add, Mul};

use crate::backend::Backend as BackendTrait;

#[derive(Clone, Debug)]
pub struct Backend {}

impl<T: Clone> BackendTrait<T> for Backend {
    type Index = usize;
    type Dimension = usize;
    type Tensor0Repr = T;
    type Tensor1Repr = Vec<T>;
    type Tensor2Repr = Vec<Vec<T>>;

    fn scalar_zero() -> Self::Tensor0Repr
    where
        T: From<u8>,
    {
        0.into()
    }

    fn scalar_one() -> Self::Tensor0Repr
    where
        T: From<u8>,
    {
        1.into()
    }

    fn vector_zeros(d0: usize) -> Self::Tensor1Repr
    where
        T: From<u8> + Copy,
    {
        vec![0.into(); d0]
    }

    fn vector_ones(d0: usize) -> Self::Tensor1Repr
    where
        T: From<u8> + Copy,
    {
        vec![1.into(); d0]
    }

    fn matrix_zeros(d0: usize, d1: usize) -> Self::Tensor2Repr
    where
        T: From<u8> + Copy,
    {
        vec![vec![0.into(); d1]; d0]
    }

    fn matrix_ones(d0: usize, d1: usize) -> Self::Tensor2Repr
    where
        T: From<u8> + Copy,
    {
        vec![vec![1.into(); d1]; d0]
    }

    fn matrix_transpose(a: Self::Tensor2Repr) -> Self::Tensor2Repr
    where
        T: From<u8> + Copy,
    {
        let mut result = Self::matrix_zeros(a[0].len(), a.len());
        for i in 0..a.len() {
            for j in 0..a[0].len() {
                result[j][i] = a[i][j];
            }
        }
        result
    }

    fn matrix_identity(d: usize) -> Self::Tensor2Repr
    where
        T: From<u8> + Copy,
    {
        let mut matrix = Self::matrix_zeros(d, d);
        for i in 0..d {
            matrix[i][i] = 1.into();
        }
        matrix
    }

    fn scalar_scalar_add(a: Self::Tensor0Repr, b: Self::Tensor0Repr) -> Self::Tensor0Repr
    where
        T: Add<Output = T>,
    {
        a + b
    }

    fn vector_scalar_add(a: Self::Tensor1Repr, b: Self::Tensor0Repr) -> Self::Tensor1Repr
    where
        T: Add<Output = T> + Copy,
    {
        a.into_iter().map(|a| a + b).collect()
    }

    fn vector_vector_add(a: Self::Tensor1Repr, b: Self::Tensor1Repr) -> Self::Tensor1Repr
    where
        T: Add<Output = T>,
    {
        assert_eq!(a.len(), b.len());
        a.into_iter()
            .zip(b.into_iter())
            .map(|(a, b)| a + b)
            .collect()
    }

    fn matrix_scalar_add(a: Self::Tensor2Repr, b: Self::Tensor0Repr) -> Self::Tensor2Repr
    where
        T: Add<Output = T> + Copy,
    {
        a.into_iter()
            .map(|a| a.into_iter().map(|a| a + b).collect())
            .collect()
    }

    fn matrix_vector_add(
        a: Self::Tensor2Repr,
        b: Self::Tensor1Repr,
        along: Self::Dimension,
    ) -> Self::Tensor2Repr
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

    fn matrix_matrix_add(a: Self::Tensor2Repr, b: Self::Tensor2Repr) -> Self::Tensor2Repr
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

    fn dot(a: Self::Tensor1Repr, b: Self::Tensor1Repr) -> Self::Tensor0Repr
    where
        T: Add<Output = T> + Mul<Output = T> + From<u8>,
    {
        a.into_iter()
            .zip(b.into_iter())
            .fold(0.into(), |acc, (a, b)| acc + a * b)
    }

    fn matmul(a: Self::Tensor2Repr, b: Self::Tensor2Repr) -> Self::Tensor2Repr
    where
        T: Add<Output = T> + Mul<Output = T> + From<u8> + Copy,
    {
        let mut result = Self::matrix_zeros(a.len(), b[0].len());
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
        let zero: u8 = Backend::scalar_zero();
        assert_eq!(zero, 0);
    }

    #[test]
    fn test_scalar_one() {
        let one: u8 = Backend::scalar_one();
        assert_eq!(one, 1);
    }

    #[test]
    fn test_vector_zeros() {
        let zeros: Vec<u8> = Backend::vector_zeros(3);
        assert_eq!(zeros, vec![0, 0, 0]);
    }

    #[test]
    fn test_vector_ones() {
        let ones: Vec<u8> = Backend::vector_ones(3);
        assert_eq!(ones, vec![1, 1, 1]);
    }

    #[test]
    fn test_matrix_zeros() {
        let zeros: Vec<Vec<u8>> = Backend::matrix_zeros(3, 3);
        assert_eq!(zeros, vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]]);
    }

    #[test]
    fn test_matrix_ones() {
        let ones: Vec<Vec<u8>> = Backend::matrix_ones(3, 3);
        assert_eq!(ones, vec![vec![1, 1, 1], vec![1, 1, 1], vec![1, 1, 1]]);
    }

    #[test]
    fn test_matrix_identity() {
        let eye: Vec<Vec<u8>> = Backend::matrix_identity(3);
        assert_eq!(eye, vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]);
    }

    #[test]
    fn test_scalar_scalar_add() {
        assert_eq!(Backend::scalar_scalar_add(1, 2), 3);
    }

    #[test]
    fn test_vector_scalar_add() {
        assert_eq!(Backend::vector_scalar_add(vec![1, 2, 3], 2), vec![3, 4, 5]);
    }

    #[test]
    fn test_vector_vector_add() {
        assert_eq!(
            Backend::vector_vector_add(vec![1, 2, 3], vec![4, 5, 6]),
            vec![5, 7, 9]
        );
    }

    #[test]
    fn test_matrix_scalar_add() {
        assert_eq!(
            Backend::matrix_scalar_add(vec![vec![1, 2, 3], vec![4, 5, 6]], 2),
            vec![vec![3, 4, 5], vec![6, 7, 8]]
        );
    }

    #[test]
    fn test_matrix_vector_add_along_0() {
        assert_eq!(
            Backend::matrix_vector_add(vec![vec![1, 2, 3], vec![4, 5, 6]], vec![2, 3], 0),
            vec![vec![3, 4, 5], vec![7, 8, 9]]
        );
    }

    #[test]
    fn test_matrix_vector_add_along_1() {
        assert_eq!(
            Backend::matrix_vector_add(vec![vec![1, 2, 3], vec![4, 5, 6]], vec![2, 3, 4], 1),
            vec![vec![3, 5, 7], vec![6, 8, 10]]
        );
    }

    #[test]
    fn test_matrix_matrix_add() {
        assert_eq!(
            Backend::matrix_matrix_add(
                vec![vec![1, 2, 3], vec![4, 5, 6]],
                vec![vec![2, 3, 4], vec![5, 6, 7]]
            ),
            vec![vec![3, 5, 7], vec![9, 11, 13]]
        );
    }

    #[test]
    fn test_dot() {
        assert_eq!(Backend::dot(vec![1, 2, 3], vec![4, 5, 6]), 32);
    }

    #[test]
    fn test_matmul() {
        assert_eq!(
            Backend::matmul(
                vec![vec![1, 2, 3], vec![4, 5, 6]],
                vec![vec![2, 3], vec![4, 5], vec![6, 7]]
            ),
            vec![vec![28, 34], vec![64, 79]]
        );
    }
}
