use tensr::tensor::{Matrix, Scalar, Tensor, Vector};

fn main() {
    let a = Matrix::ones();
    let b = Matrix::identity();
    let c = a.matmul(b);
    let d = c + Matrix::<f32, 3, 3>::identity();

    let e = Matrix::<f32, 3, 5>::ones();
    // let g = f + Matrix::<f32, 5, 3>::ones();
    println!("{:?}", e.shape());
}
