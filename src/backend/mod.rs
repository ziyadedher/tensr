use std::ops::{Add, Mul};

use crate::tensor::{Dimension, Index};

pub mod basic;

pub type AutoSelectBackend = basic::Backend;

pub trait Backend<T> {
    type Index: From<Index>;
    type Dimension: From<Dimension>;
    type ScalarRepr;
    type VectorRepr;
    type MatrixRepr;

    fn scalar_zero() -> Self::ScalarRepr
    where
        T: From<u8>;
    fn scalar_one() -> Self::ScalarRepr
    where
        T: From<u8>;
    fn vector_zeros(d0: Self::Index) -> Self::VectorRepr
    where
        T: From<u8> + Copy;
    fn vector_ones(d0: Self::Index) -> Self::VectorRepr
    where
        T: From<u8> + Copy;
    fn matrix_zeros(d0: Self::Index, d1: Self::Index) -> Self::MatrixRepr
    where
        T: From<u8> + Copy;
    fn matrix_ones(d0: Self::Index, d1: Self::Index) -> Self::MatrixRepr
    where
        T: From<u8> + Copy;
    fn matrix_identity(d: Self::Index) -> Self::MatrixRepr
    where
        T: From<u8> + Copy;

    fn matrix_transpose(a: Self::MatrixRepr) -> Self::MatrixRepr
    where
        T: From<u8> + Copy;

    fn scalar_scalar_add(a: Self::ScalarRepr, b: Self::ScalarRepr) -> Self::ScalarRepr
    where
        T: Add<Output = T>;
    fn vector_scalar_add(a: Self::VectorRepr, b: Self::ScalarRepr) -> Self::VectorRepr
    where
        T: Add<Output = T> + Copy;
    fn vector_vector_add(a: Self::VectorRepr, b: Self::VectorRepr) -> Self::VectorRepr
    where
        T: Add<Output = T>;
    fn matrix_scalar_add(a: Self::MatrixRepr, b: Self::ScalarRepr) -> Self::MatrixRepr
    where
        T: Add<Output = T> + Copy;
    fn matrix_vector_add(
        a: Self::MatrixRepr,
        b: Self::VectorRepr,
        along: Self::Dimension,
    ) -> Self::MatrixRepr
    where
        T: Add<Output = T> + Copy;
    fn matrix_matrix_add(a: Self::MatrixRepr, b: Self::MatrixRepr) -> Self::MatrixRepr
    where
        T: Add<Output = T>;

    fn dot(a: Self::VectorRepr, b: Self::VectorRepr) -> Self::ScalarRepr
    where
        T: Add<Output = T> + Mul<Output = T> + From<u8>;
    fn matmul(a: Self::MatrixRepr, b: Self::MatrixRepr) -> Self::MatrixRepr
    where
        T: Add<Output = T> + Mul<Output = T> + From<u8> + Copy;
}
