pub mod tensor0;
pub mod tensor1;
pub mod tensor2;

pub use tensor0::Tensor0 as Scalar;
pub use tensor1::Tensor1 as Vector;
pub use tensor2::Tensor2 as Matrix;

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
