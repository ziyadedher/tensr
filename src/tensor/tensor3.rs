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
