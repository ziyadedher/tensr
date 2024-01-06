use std::ops::{Add, Mul};

use crate::{
    backend::{AutoSelectBackend, Backend},
    tensor::{Scalar, Tensor},
};

#[derive(Clone, Debug)]
pub struct Vector<T, const D0: usize, B: Backend<T> = AutoSelectBackend> {
    pub(crate) repr: B::VectorRepr,
    pub(crate) shape: <Vector<T, D0, B> as Tensor>::Shape,
}

impl<T, const D0: usize, B: Backend<T>> Vector<T, D0, B> {
    pub fn dot(self, other: Vector<T, D0, B>) -> Scalar<T, B>
    where
        T: From<u8> + Add<Output = T> + Mul<Output = T>,
    {
        assert_eq!(self.shape, other.shape);

        Scalar {
            repr: B::dot(self.repr, other.repr),
            shape: (),
        }
    }
}

impl<T, const D0: usize, B: Backend<T>> Tensor for Vector<T, D0, B> {
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
            repr: B::vector_zeros(D0.into()),
            shape: D0,
        }
    }

    fn ones() -> Self
    where
        Self::DataType: From<u8> + Copy,
    {
        Self {
            repr: B::vector_ones(D0.into()),
            shape: D0,
        }
    }
}

impl<T, const D0: usize, B: Backend<T>> Add<Scalar<T, B>> for Vector<T, D0, B>
where
    T: Add<Output = T> + Copy,
{
    type Output = Self;

    fn add(self, other: Scalar<T, B>) -> Self {
        Self {
            repr: B::vector_scalar_add(self.repr, other.repr),
            shape: D0,
        }
    }
}

impl<T, const D0: usize, B: Backend<T>> Add for Vector<T, D0, B>
where
    T: Add<Output = T>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            repr: B::vector_vector_add(self.repr, other.repr),
            shape: D0,
        }
    }
}
