use std::ops::Add;

use crate::{backend::Backend, tensor::Tensor};

#[derive(Clone, Debug)]
pub struct Scalar<B: Backend<T>, T> {
    pub(crate) repr: B::ScalarRepr,
    pub(crate) shape: <Scalar<B, T> as Tensor>::Shape,
}

impl<B: Backend<T>, T> Tensor for Scalar<B, T> {
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

impl<B: Backend<T>, T> Add for Scalar<B, T>
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
