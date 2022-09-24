/// Module for matrix operations

use std::ops::{Add, AddAssign, Mul, Sub};

#[derive(Eq, PartialEq, Debug, Clone, Default)]
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

impl<T : Add<T, Output = T> + Default + Copy> Add for Matrix<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        if self.width != other.width || self.height != other.height {
            panic!("Cannot add two matrices, incompatible dimensions.")
        }

        let mut sum = Matrix::create_matrix(self.width, self.height);

        for y in 0..self.height {
            for x in 0..other.width {
                sum.set(y, x, self.get(y, x) + other.get(y, x));
            }
        }

        sum
    }
}


impl<T : Sub<T, Output = T> + Default + Copy> Sub for Matrix<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        if self.width != other.width || self.height != other.height {
            panic!("Cannot subtract two matrices, incompatible dimensions.")
        }

        let mut diff = Matrix::create_matrix(self.width, self.height);

        for y in 0..self.height {
            for x in 0..other.width {
                diff.set(y, x, self.get(y, x) - other.get(y, x));
            }
        }

        diff
    }
}


impl<T : Mul<T, Output = T> + Add<T, Output = T> + Default + Copy + AddAssign> Mul for Matrix<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        if self.width != other.height {
            panic!("Cannot multiply two matrices, incompatible dimensions.")
        }

        let mut prod = Matrix::create_matrix(self.height, other.width);

        for y in 0..self.height {
            for x in 0..self.width {

                let mut sum = T::default();

                for k in 0..self.height {
                    sum += self.get(y, k) + other.get(x, k);
                }

                prod.set(y, x, sum)
            }
        }

        prod
    }
}
