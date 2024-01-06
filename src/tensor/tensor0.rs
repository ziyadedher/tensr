use std::ops::Add;

use crate::{backend::Backend, tensor::Tensor};

#[derive(Clone, Debug)]
pub struct Tensor0<T, B: Backend<T>> {
    pub(crate) repr: B::Tensor0Repr,
    pub(crate) shape: <Tensor0<T, B> as Tensor>::Shape,
}

impl<T, B: Backend<T>> Tensor for Tensor0<T, B> {
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

impl<T, B: Backend<T>> Tensor0<T, B> {
    pub fn permute(self) -> Self {
        self
    }
}

impl<T, B: Backend<T>> Add for Tensor0<T, B>
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
