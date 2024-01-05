use tensr::tensor::{Matrix, Scalar, Tensor, Vector};

fn main() {
    let a = Matrix::ones();
    let b = Matrix::identity();
    let c = a.matmul(b);
    let d = c + Matrix::<f32, 2, 2>::identity();

    let v = Vector::ones();
    let m = Matrix::ones();
    let r = (m.transpose() + v).transpose();
    let q = r + d;

    println!("{:?}", q);
}
