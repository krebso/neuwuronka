mod matrix;

use crate::matrix::Matrix;

fn test() {
    let mut m = Matrix::<i32>::create_matrix(2, 2);

    assert_eq!(m, Matrix::<i32>::create_matrix(2, 2));

    m.set(0, 0, 1);

    assert_ne!(m, Matrix::<i32>::create_matrix(2, 2));

    assert_eq!(m.get(0, 0), 1);

    assert_eq!(m.clone() - m.clone(), Matrix::<i32>::create_matrix(2, 2));

    let mut add = m.clone();
    add.set(0, 0, 2);

    assert_eq!( m.clone() + m.clone(), add);
}

fn main() {
    test();
    println!("All tests passed!")
}
