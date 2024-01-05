pub mod matrix;
pub mod scalar;
pub mod vector;

pub use matrix::Matrix;
pub use scalar::Scalar;
pub use vector::Vector;

pub trait Tensor: Sized {
    type Shape;
    type DataType;
    type Transpose: Tensor;

    fn shape(&self) -> Self::Shape;
    fn transpose(self) -> Self::Transpose
    where
        Self::DataType: From<u8> + Copy;

    fn zeros() -> Self
    where
        Self::DataType: From<u8> + Copy;
    fn ones() -> Self
    where
        Self::DataType: From<u8> + Copy;
}

pub type Index = usize;
pub type Dimension = usize;
