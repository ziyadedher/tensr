use std::ops::{Add, Mul};

use crate::{
    backend::{AutoSelectBackend, Backend},
    tensor::{Scalar, Tensor},
};

#[derive(Clone, Debug)]
pub struct Tensor1<T, const D0: usize, B: Backend<T> = AutoSelectBackend> {
    pub(crate) repr: B::T1Repr,
    pub(crate) shape: <Tensor1<T, D0, B> as Tensor>::Shape,
}

impl<T, const D0: usize, B: Backend<T>> Tensor1<T, D0, B> {
    pub fn dot(self, other: Tensor1<T, D0, B>) -> Scalar<T, B>
    where
        T: From<u8> + Add<Output = T> + Mul<Output = T>,
    {
        assert_eq!(self.shape, other.shape);

        Scalar {
            repr: B::t1_t1_dot(self.repr, other.repr),
            shape: (),
        }
    }
}

impl<T, const D0: usize, B: Backend<T>> Tensor for Tensor1<T, D0, B> {
    type Shape = usize;
    type DataType = T;

    fn shape(&self) -> Self::Shape {
        self.shape
    }

    fn zeros() -> Self
    where
        Self::DataType: From<u8> + Copy,
    {
        Self {
            repr: B::t1_zeros(D0.into()),
            shape: D0,
        }
    }

    fn ones() -> Self
    where
        Self::DataType: From<u8> + Copy,
    {
        Self {
            repr: B::t1_ones(D0.into()),
            shape: D0,
        }
    }
}

impl<T, const D0: usize, B: Backend<T>> Tensor1<T, D0, B> {
    pub const fn construct_shape(d0: usize) -> (usize,) {
        (d0,)
    }

    pub const fn calculate_permute((p0,): (usize,), (d0,): (usize,), i: usize) -> usize {
        match (p0, i) {
            (0, 0) => d0,
            (0, 1) => d0,
            _ => panic!("improper permute"),
        }
    }

    pub fn permute<const P0: usize>(
        self,
    ) -> Tensor1<
        T,
        { Self::calculate_permute(Self::construct_shape(P0), Self::construct_shape(D0), 0) },
        B,
    > {
        Tensor1 {
            repr: self.repr,
            shape: Self::calculate_permute(Self::construct_shape(P0), Self::construct_shape(D0), 0),
        }
    }
}

impl<T, const D0: usize, B: Backend<T>> Add<Scalar<T, B>> for Tensor1<T, D0, B>
where
    T: Add<Output = T> + Copy,
{
    type Output = Self;

    fn add(self, other: Scalar<T, B>) -> Self {
        Self {
            repr: B::t1_t0_add(self.repr, other.repr),
            shape: D0,
        }
    }
}

impl<T, const D0: usize, B: Backend<T>> Add for Tensor1<T, D0, B>
where
    T: Add<Output = T>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            repr: B::t1_t1_add(self.repr, other.repr),
            shape: D0,
        }
    }
}
