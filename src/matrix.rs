/// Module for matrix operations

#[derive(Eq, PartialEq)]
pub struct Matrix<T> {
    width: usize,
    height: usize,
    _matrix: Vec<Vec<T>>,
}

impl<T: Default + Copy> Matrix<T> {
    pub fn create_matrix(width: usize, height: usize) -> Matrix<T> {
        let mut m = Matrix::<T> {
            width,
            height,
            _matrix: Vec::with_capacity(height),
        };

        for h in 0..height {
            m._matrix.push(Vec::with_capacity(width));
            for _ in 0..width {
                m._matrix[h].push(T::default());
            }
        }

        m
    }

    pub fn set(&mut self, x: usize, y: usize, value: T) {
        self._matrix[y][x] = value;
    }

    pub fn get(&self, x: usize, y: usize) -> T {
        self._matrix[y][x]
    }
}
