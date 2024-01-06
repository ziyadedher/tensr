use std::ops::{Add, Mul};

use crate::tensor::{Dimension, Index};

pub mod basic;

pub type AutoSelectBackend = basic::Backend;

pub trait Backend<T> {
    type Index: From<Index>;
    type Dimension: From<Dimension>;
    type Tensor0Repr;
    type Tensor1Repr;
    type Tensor2Repr;

    fn scalar_zero() -> Self::Tensor0Repr
    where
        T: From<u8>;
    fn scalar_one() -> Self::Tensor0Repr
    where
        T: From<u8>;
    fn vector_zeros(d0: Self::Index) -> Self::Tensor1Repr
    where
        T: From<u8> + Copy;
    fn vector_ones(d0: Self::Index) -> Self::Tensor1Repr
    where
        T: From<u8> + Copy;
    fn matrix_zeros(d0: Self::Index, d1: Self::Index) -> Self::Tensor2Repr
    where
        T: From<u8> + Copy;
    fn matrix_ones(d0: Self::Index, d1: Self::Index) -> Self::Tensor2Repr
    where
        T: From<u8> + Copy;
    fn matrix_identity(d: Self::Index) -> Self::Tensor2Repr
    where
        T: From<u8> + Copy;

    fn matrix_transpose(a: Self::Tensor2Repr) -> Self::Tensor2Repr
    where
        T: From<u8> + Copy;

    fn scalar_scalar_add(a: Self::Tensor0Repr, b: Self::Tensor0Repr) -> Self::Tensor0Repr
    where
        T: Add<Output = T>;
    fn vector_scalar_add(a: Self::Tensor1Repr, b: Self::Tensor0Repr) -> Self::Tensor1Repr
    where
        T: Add<Output = T> + Copy;
    fn vector_vector_add(a: Self::Tensor1Repr, b: Self::Tensor1Repr) -> Self::Tensor1Repr
    where
        T: Add<Output = T>;
    fn matrix_scalar_add(a: Self::Tensor2Repr, b: Self::Tensor0Repr) -> Self::Tensor2Repr
    where
        T: Add<Output = T> + Copy;
    fn matrix_vector_add(
        a: Self::Tensor2Repr,
        b: Self::Tensor1Repr,
        along: Self::Dimension,
    ) -> Self::Tensor2Repr
    where
        T: Add<Output = T> + Copy;
    fn matrix_matrix_add(a: Self::Tensor2Repr, b: Self::Tensor2Repr) -> Self::Tensor2Repr
    where
        T: Add<Output = T>;

    fn dot(a: Self::Tensor1Repr, b: Self::Tensor1Repr) -> Self::Tensor0Repr
    where
        T: Add<Output = T> + Mul<Output = T> + From<u8>;
    fn matmul(a: Self::Tensor2Repr, b: Self::Tensor2Repr) -> Self::Tensor2Repr
    where
        T: Add<Output = T> + Mul<Output = T> + From<u8> + Copy;
}
