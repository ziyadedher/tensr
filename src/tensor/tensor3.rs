use std::ops::{Add, Mul};

use crate::{
    backend::{AutoSelectBackend, Backend},
    tensor::{Scalar, Tensor, Vector},
};

#[derive(Clone, Debug)]
pub struct Tensor3<
    T,
    const D0: usize,
    const D1: usize,
    const D2: usize,
    B: Backend<T> = AutoSelectBackend,
> {
    pub(crate) repr: B::T3Repr,
    pub(crate) shape: <Tensor3<T, D0, D1, D2, B> as Tensor>::Shape,
}

impl<T, const D0: usize, const D1: usize, const D2: usize, B: Backend<T>>
    Tensor3<T, D0, D1, D2, B>
{
    pub const fn construct_shape(d0: usize, d1: usize, d2: usize) -> (usize, usize, usize) {
        (d0, d1, d2)
    }

    pub const fn calculate_permute(
        (p0, p1, p2): (usize, usize, usize),
        (d0, d1, d2): (usize, usize, usize),
        i: usize,
    ) -> usize {
        match (p0, p1, p2, i) {
            (0, 1, 2, 0) => d0,
            (0, 1, 2, 1) => d1,
            (0, 1, 2, 2) => d2,
            (0, 2, 1, 0) => d0,
            (0, 2, 1, 1) => d2,
            (0, 2, 1, 2) => d1,
            (1, 0, 2, 0) => d1,
            (1, 0, 2, 1) => d0,
            (1, 0, 2, 2) => d2,
            (1, 2, 0, 0) => d2,
            (1, 2, 0, 1) => d0,
            (1, 2, 0, 2) => d1,
            (2, 0, 1, 0) => d1,
            (2, 0, 1, 1) => d2,
            (2, 0, 1, 2) => d0,
            (2, 1, 0, 0) => d2,
            (2, 1, 0, 1) => d1,
            (2, 1, 0, 2) => d0,
            _ => panic!("improper permute"),
        }
    }

    pub fn permute<const P0: usize, const P1: usize, const P2: usize>(
        self,
    ) -> Tensor3<
        T,
        {
            Self::calculate_permute(
                Self::construct_shape(P0, P1, P2),
                Self::construct_shape(D0, D1, D2),
                0,
            )
        },
        {
            Self::calculate_permute(
                Self::construct_shape(P0, P1, P2),
                Self::construct_shape(D0, D1, D2),
                1,
            )
        },
        {
            Self::calculate_permute(
                Self::construct_shape(P0, P1, P2),
                Self::construct_shape(D0, D1, D2),
                2,
            )
        },
        B,
    >
    where
        T: From<u8> + Copy,
    {
        Tensor3 {
            repr: B::t3_permute(self.repr, (P0.into(), P1.into(), P2.into())),
            shape: (
                Self::calculate_permute((P0, P1, P2), (D0, D1, D2), 0),
                Self::calculate_permute((P0, P1, P2), (D0, D1, D2), 1),
                Self::calculate_permute((P0, P1, P2), (D0, D1, D2), 2),
            ),
        }
    }
}

impl<T, const D0: usize, const D1: usize, const D2: usize, B: Backend<T>> Tensor
    for Tensor3<T, D0, D1, D2, B>
{
    type Shape = (usize, usize, usize);
    type DataType = T;

    fn shape(&self) -> Self::Shape {
        self.shape
    }

    fn zeros() -> Self
    where
        Self::DataType: From<u8> + Copy,
    {
        Self {
            repr: B::t3_zeros(D0.into(), D1.into(), D2.into()),
            shape: (D0, D1, D2),
        }
    }

    fn ones() -> Self
    where
        Self::DataType: From<u8> + Copy,
    {
        Self {
            repr: B::t3_ones(D0.into(), D1.into(), D2.into()),
            shape: (D0, D1, D2),
        }
    }
}
