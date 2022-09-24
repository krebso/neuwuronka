mod matrix;

use crate::matrix::Matrix;

fn test() {
    let mut m = Matrix::create_matrix(2, 2);

    assert_eq!(m, Matrix::create_matrix(2, 2));

    m.set(0, 0, 1f32);

    assert_ne!(m, Matrix::create_matrix(2, 2));

    assert_eq!(m.get(0, 0), 1f32);

    assert_eq!(m.clone() - m.clone(), Matrix::create_matrix(2, 2));

    let mut add = m.clone();
    add.set(0, 0, 2f32);

    assert_eq!( m.clone() + m.clone(), add);
}

fn main() {
    test();
    println!("All tests passed!")
}
