use std::ops::{Add, Mul};

use crate::tensor::{Dimension, Index};

pub mod basic;

pub type AutoSelectBackend = basic::Backend;

pub trait Backend<T> {
    type Index: From<Index>;
    type Dimension: From<Dimension>;
    type T0Repr;
    type T1Repr;
    type T2Repr;
    type T3Repr;

    fn t0_zero() -> Self::T0Repr
    where
        T: From<u8>;
    fn t0_one() -> Self::T0Repr
    where
        T: From<u8>;
    fn t1_zeros(d0: Self::Index) -> Self::T1Repr
    where
        T: From<u8> + Copy;
    fn t1_ones(d0: Self::Index) -> Self::T1Repr
    where
        T: From<u8> + Copy;
    fn t2_zeros(d0: Self::Index, d1: Self::Index) -> Self::T2Repr
    where
        T: From<u8> + Copy;
    fn t2_ones(d0: Self::Index, d1: Self::Index) -> Self::T2Repr
    where
        T: From<u8> + Copy;
    fn t3_ones(d0: Self::Index, d1: Self::Index, d2: Self::Index) -> Self::T3Repr
    where
        T: From<u8> + Copy;
    fn t3_zeros(d0: Self::Index, d1: Self::Index, d2: Self::Index) -> Self::T3Repr
    where
        T: From<u8> + Copy;

    fn t2_identity(d: Self::Index) -> Self::T2Repr
    where
        T: From<u8> + Copy;
    fn t2_transpose(a: Self::T2Repr) -> Self::T2Repr
    where
        T: From<u8> + Copy;
    fn t3_permute(
        a: Self::T3Repr,
        p: (Self::Dimension, Self::Dimension, Self::Dimension),
    ) -> Self::T3Repr
    where
        T: From<u8> + Copy;

    fn t0_t0_add(a: Self::T0Repr, b: Self::T0Repr) -> Self::T0Repr
    where
        T: Add<Output = T>;
    fn t1_t0_add(a: Self::T1Repr, b: Self::T0Repr) -> Self::T1Repr
    where
        T: Add<Output = T> + Copy;
    fn t1_t1_add(a: Self::T1Repr, b: Self::T1Repr) -> Self::T1Repr
    where
        T: Add<Output = T>;
    fn t2_t0_add(a: Self::T2Repr, b: Self::T0Repr) -> Self::T2Repr
    where
        T: Add<Output = T> + Copy;
    fn t2_t1_add(a: Self::T2Repr, b: Self::T1Repr, along: Self::Dimension) -> Self::T2Repr
    where
        T: Add<Output = T> + Copy;
    fn t2_t2_add(a: Self::T2Repr, b: Self::T2Repr) -> Self::T2Repr
    where
        T: Add<Output = T>;

    fn t1_t1_dot(a: Self::T1Repr, b: Self::T1Repr) -> Self::T0Repr
    where
        T: Add<Output = T> + Mul<Output = T> + From<u8>;
    fn t2_t2_matmul(a: Self::T2Repr, b: Self::T2Repr) -> Self::T2Repr
    where
        T: Add<Output = T> + Mul<Output = T> + From<u8> + Copy;
}
