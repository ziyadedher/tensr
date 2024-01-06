use std::ops::{Add, Mul};

use crate::{
    backend::{AutoSelectBackend, Backend},
    tensor::{Scalar, Tensor, Vector},
};

#[derive(Clone, Debug)]
pub struct Matrix<T, const D0: usize, const D1: usize, B: Backend<T> = AutoSelectBackend> {
    pub(crate) repr: B::MatrixRepr,
    pub(crate) shape: <Matrix<T, D0, D1, B> as Tensor>::Shape,
}

impl<T, const D: usize, B: Backend<T>> Matrix<T, D, D, B> {
    pub fn identity() -> Self
    where
        T: From<u8> + Copy,
    {
        Self {
            repr: B::matrix_identity(D.into()),
            shape: (D, D),
        }
    }
}

impl<T, const D0: usize, const D1: usize, B: Backend<T>> Matrix<T, D0, D1, B> {
    pub const fn construct_shape(d0: usize, d1: usize) -> (usize, usize) {
        (d0, d1)
    }

    pub const fn calculate_permute(
        (p0, p1): (usize, usize),
        (d0, d1): (usize, usize),
        i: usize,
    ) -> usize {
        match (p0, p1, i) {
            (0, 1, 0) => d0,
            (0, 1, 1) => d1,
            (1, 0, 0) => d1,
            (1, 0, 1) => d0,
            _ => panic!("improper permute"),
        }
    }

    pub fn permute<const P0: usize, const P1: usize>(
        self,
    ) -> Matrix<
        T,
        {
            Self::calculate_permute(
                Self::construct_shape(P0, P1),
                Self::construct_shape(D0, D1),
                0,
            )
        },
        {
            Self::calculate_permute(
                Self::construct_shape(P0, P1),
                Self::construct_shape(D0, D1),
                1,
            )
        },
        B,
    >
    where
        T: From<u8> + Copy,
    {
        Matrix {
            repr: B::matrix_transpose(self.repr),
            shape: (
                Self::calculate_permute((P0, P1), (D0, D1), 0),
                Self::calculate_permute((P0, P1), (D0, D1), 1),
            ),
        }
    }
}

impl<T, const D0: usize, const D1: usize, B: Backend<T>> Matrix<T, D0, D1, B> {
    pub fn matmul<const OD1: usize>(self, other: Matrix<T, D1, OD1, B>) -> Matrix<T, D0, OD1, B>
    where
        T: Add<Output = T> + Mul<Output = T> + From<u8> + Copy,
    {
        assert_eq!(self.shape.1, other.shape.0);

        Matrix {
            repr: B::matmul(self.repr, other.repr),
            shape: (D0, OD1),
        }
    }
}

impl<T, const D0: usize, const D1: usize, B: Backend<T>> Tensor for Matrix<T, D0, D1, B> {
    type Shape = (usize, usize);
    type DataType = T;

    fn shape(&self) -> Self::Shape {
        self.shape
    }

    fn zeros() -> Self
    where
        Self::DataType: From<u8> + Copy,
    {
        Self {
            repr: B::matrix_zeros(D0.into(), D1.into()),
            shape: (D0, D1),
        }
    }

    fn ones() -> Self
    where
        Self::DataType: From<u8> + Copy,
    {
        Self {
            repr: B::matrix_ones(D0.into(), D1.into()),
            shape: (D0, D1),
        }
    }
}

impl<T, const D0: usize, const D1: usize, B: Backend<T>> Add<Scalar<T, B>> for Matrix<T, D0, D1, B>
where
    T: Add<Output = T> + Copy,
{
    type Output = Self;

    fn add(self, other: Scalar<T, B>) -> Self {
        Self {
            repr: B::matrix_scalar_add(self.repr, other.repr),
            shape: self.shape,
        }
    }
}

/// FIXME: ideally, we'd be able to add a vector to the matrix even if the final dimensions don't match,
/// but we can't do that right now in a way that maintains the type system's safety. So we rely on folks doing
/// something like `matrix.transpose().add(vector).transpose()` instead.
impl<T, const D0: usize, const D1: usize, B: Backend<T>> Add<Vector<T, D1, B>>
    for Matrix<T, D0, D1, B>
where
    T: Add<Output = T> + Copy,
{
    type Output = Self;

    fn add(self, other: Vector<T, D1, B>) -> Self {
        Self {
            repr: B::matrix_vector_add(self.repr, other.repr, 1.into()),
            shape: self.shape,
        }
    }
}

impl<T, const D0: usize, const D1: usize, B: Backend<T>> Add for Matrix<T, D0, D1, B>
where
    T: Add<Output = T>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(self.shape, other.shape);

        Self {
            repr: B::matrix_matrix_add(self.repr, other.repr),
            shape: self.shape,
        }
    }
}
