use std::ops::Add;

use crate::{backend::Backend, tensor::Tensor};

#[derive(Clone, Debug)]
pub struct Scalar<T, B: Backend<T>> {
    pub(crate) repr: B::ScalarRepr,
    pub(crate) shape: <Scalar<T, B> as Tensor>::Shape,
}

impl<T, B: Backend<T>> Tensor for Scalar<T, B> {
    type Shape = ();
    type DataType = T;

    fn shape(&self) -> Self::Shape {
        self.shape
    }

    fn zeros() -> Self
    where
        Self::DataType: From<u8> + Copy,
    {
        Self {
            repr: B::scalar_zero(),
            shape: (),
        }
    }

    fn ones() -> Self
    where
        Self::DataType: From<u8> + Copy,
    {
        Self {
            repr: B::scalar_one(),
            shape: (),
        }
    }
}

impl<T, B: Backend<T>> Scalar<T, B> {
    pub fn permute(self) -> Self {
        self
    }
}

impl<T, B: Backend<T>> Add for Scalar<T, B>
where
    T: Add<Output = T>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            repr: B::scalar_scalar_add(self.repr, other.repr),
            shape: (),
        }
    }
}
