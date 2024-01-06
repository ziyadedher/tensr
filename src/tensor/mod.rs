pub mod matrix;
pub mod scalar;
pub mod vector;

pub use matrix::Matrix;
pub use scalar::Scalar;
pub use vector::Vector;

pub trait Tensor {
    type Shape;
    type DataType;

    fn shape(&self) -> Self::Shape;

    fn zeros() -> Self
    where
        Self::DataType: From<u8> + Copy;
    fn ones() -> Self
    where
        Self::DataType: From<u8> + Copy;
}

pub type Index = usize;
pub type Dimension = usize;
