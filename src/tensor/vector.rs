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

impl<B: Backend<T>, T, const D0: usize> Vector<T, D0, B> {
    pub fn dot(self, other: Vector<T, D0, B>) -> Scalar<B, T>
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

impl<B: Backend<T>, T, const D0: usize> Tensor for Vector<T, D0, B> {
    type Shape = usize;
    type DataType = T;
    type Transpose = Self;

    fn shape(&self) -> Self::Shape {
        self.shape
    }

    fn transpose(self) -> Self {
        self
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

impl<B: Backend<T>, T, const D0: usize> Add<Scalar<B, T>> for Vector<T, D0, B>
where
    T: Add<Output = T> + Copy,
{
    type Output = Self;

    fn add(self, other: Scalar<B, T>) -> Self {
        Self {
            repr: B::vector_scalar_add(self.repr, other.repr),
            shape: D0,
        }
    }
}

impl<B: Backend<T>, T, const D0: usize> Add for Vector<T, D0, B>
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
