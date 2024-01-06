use tensr::tensor::{Matrix, Scalar, Tensor, Vector};

fn main() {
    let a = Matrix::ones();
    let b = Matrix::identity();
    let c = a.matmul(b);
    let d = c + Matrix::<f32, 3, 3>::identity();

    let e = Matrix::<f32, 3, 5>::ones();
    println!("{:?}", e.shape());

    let a1 = Matrix::<f32, 3, 5>::zeros().permute::<1, 0>();

    let b1 = Matrix::zeros();
    let c1: Matrix<f32, 5, 3> = a1 + b1;
}
