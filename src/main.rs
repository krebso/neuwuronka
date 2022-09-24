mod matrix;

use crate::matrix::Matrix;

fn test() {
    let mut m = Matrix::<i32>::create_matrix(2, 2);

    assert!(m == Matrix::<i32>::create_matrix(2, 2));

    m.set(0, 0, 1);

    assert!(m != Matrix::<i32>::create_matrix(2, 2));

    assert!(m.get(0, 0) == 1);
}

fn main() {
    test();
    println!("All tests passed!")
}
