//
// Created by Martin Krebs on 15/10/2022.
//

#ifndef NEUWURONKA_MATRIX_HPP
#define NEUWURONKA_MATRIX_HPP

#include <vector>

#include "/opt/homebrew/Cellar/libomp/15.0.6/include/omp.h"
#include "math.hpp"

constexpr size_t THREADS = 4;

template <size_t N>
struct Vector {
  static constexpr size_t size = N;
  std::vector<float> vector;

  Vector() : vector(N) {}

  void zero() {
#pragma omp parallel num_threads(THREADS)
    {
#pragma omp for
        for (size_t i = 0; i < N; ++i) vector[i] = 0.0f;
    }
  }

  float operator[](size_t i) const { return vector[i]; }

  float &operator[](size_t i) { return vector[i]; }

  Vector<N> &operator=(const Vector<N> &other) {
    vector = other.vector;
    return *this;
  }

  Vector<N> &operator+(const Vector<N> &other) {
#pragma omp parallel num_threads(THREADS)
    {
#pragma omp for
        for (size_t i = 0; i < N; i++) vector[i] += other[i];
      return *this;
    }
  }

  Vector<N> &operator+=(const Vector<N> &other) {
#pragma omp parallel num_threads(THREADS)
    {
#pragma omp for
        for (size_t i = 0; i < N; i++) vector[i] += other[i];
      return *this;
    }
  }

  Vector<N> &operator-(Vector<N> other) {
#pragma omp parallel num_threads(THREADS)
    {
#pragma omp for
        for (size_t i = 0; i < N; i++) vector[i] -= other[i];
      return *this;
    }
  }

  Vector<N> &operator-=(Vector<N> other) {
#pragma omp parallel num_threads(THREADS)
    {
#pragma omp for
        for (size_t i = 0; i < N; i++) vector[i] -= other[i];
      return *this;
    }
  }

  Vector<N> &operator*(Vector<N> other) {
#pragma omp parallel num_threads(THREADS)
    {
#pragma omp for
        for (size_t i = 0; i < N; i++) vector[i] *= other[i];
        return *this;
    }
  }

  Vector<N> &operator*=(Vector<N> other) {
#pragma omp parallel num_threads(THREADS)
      {
#pragma omp for
              for (size_t i = 0; i < N; i++) vector[i] *= other[i];
          return *this;
      }
  }

  Vector<N> &operator*=(float k) {
#pragma omp parallel num_threads(THREADS)
      {
#pragma omp for
              for (size_t i = 0; i < N; ++i) vector[i] *= k;
          return *this;
      }
  }

  Vector<N> &operator*(float k) {
#pragma omp parallel num_threads(THREADS)
      {
#pragma omp for
              for (size_t i = 0; i < N; ++i) vector[i] *= k;
          return *this;
      }
  }

  [[nodiscard]] size_t imax() const {
    size_t i = 0;
    for (size_t j = 1; j < size; ++j)
      if (vector[j] > vector[i]) i = j;
    return i;
  }
  auto to_string() const {
    std::string sep;
    std::string r = "[";
    for (auto v : vector) {
      r += sep + std::to_string(v);
      sep = ", ";
    }
    r += "]";
    return r;
  }
};

template <size_t H, size_t W>
struct Matrix {
  static constexpr size_t height = H;
  static constexpr size_t width = W;
  std::vector<float> matrix;

  Matrix() : matrix(H * W) {}

  Matrix<H, W> &operator=(const Matrix<H, W> &other) {
    matrix = other.matrix;
    return *this;
  }

  void zero() {
    for (auto &v : matrix) v = 0;
  }

  constexpr float &at(size_t h, size_t w) { return matrix[h * W + w]; }

  [[nodiscard]] constexpr const float &at(size_t h, size_t w) const {
    return matrix[h * W + w];
  }

  inline Matrix<H, W> &operator+=(const Matrix<H, W> &other) {
#pragma omp parallel num_threads(THREADS)
      {
#pragma omp for collapse(2)
          for (size_t i = 0; i < H; i++)
              for (size_t j = 0; j < W; j++) at(i, j) += other.at(i, j);
          return *this;
      }
  };

  inline Matrix<H, W> &operator-=(const Matrix<H, W> &other) {
#pragma omp parallel num_threads(THREADS)
      {
#pragma omp for collapse(2)
          for (size_t i = 0; i < H; i++)
              for (size_t j = 0; j < W; j++) at(i, j) -= other.at(i, j);
    return *this;
      }
  };

  inline auto &operator*(float k) {
#pragma omp parallel num_threads(THREADS)
      {
#pragma omp for
          for (size_t i = 0; i < H * W; ++i) matrix[i] *= k;
          return *this;
      }
  }

  inline auto &operator*=(float k) {
#pragma omp parallel num_threads(THREADS)
      {
#pragma omp for
          for (size_t i = 0; i < H * W; ++i) matrix[i] *= k;
          return *this;
      }
  }

  auto to_string() const {
    std::string sep;
    std::string res;

    for (size_t h = 0; h < height; ++h) {
      sep = "";
      for (size_t w = 0; w < width; ++w) {
        res += sep + std::to_string(this->at(h, w));
        sep = ", ";
      }
      res += "\n";
    }

    return res;
  }
};

template <typename F, typename V>
auto &map(const F &f, const V &v, V &out) {
#pragma omp parallel num_threads(THREADS)
    {
#pragma omp for
        for (size_t i = 0; i < V::size; ++i) out.vector[i] = f(v.vector[i]);
  return v;
    }
}

template <size_t S>
constexpr Vector<S> onehot(size_t i) noexcept {
  Vector<S> _onehot;
  _onehot[i] = 1.0;
  return _onehot;
}

template <size_t H, size_t W>
auto &dot_matrix_vector_transposed(const Matrix<H, W> &m, const Vector<W> &v,
                                   Vector<H> &out) {
#pragma omp parallel num_threads(THREADS)
    {
        out.zero();
#pragma omp for collapse(2)
        for (size_t h = 0; h < H; ++h)
            for (size_t w = 0; w < W; ++w) out.vector[h] += m.at(h, w) * v[w];
        return out;
    }
}

template <size_t H, size_t W>
auto &dot_vector_transposed_vector(const Vector<H> &hv, const Vector<W> &wv,
                                   Matrix<H, W> &out) {
#pragma omp parallel num_threads(THREADS)
    {
#pragma omp for collapse(2)
        for (size_t h = 0; h < H; ++h)
            for (size_t w = 0; w < W; ++w) out.at(h, w) = hv[h] * wv[w];
        return out;
    }
}

template <size_t H, size_t W>
auto &dot_matrix_transposed_vector(const Matrix<H, W> &m, const Vector<H> &v,
                                   Vector<W> &out) {
#pragma omp parallel num_threads(THREADS)
    {
        out.zero();
#pragma omp for collapse(2)
  for (size_t h = 0; h < H; ++h)
    for (size_t w = 0; w < W; ++w) out.vector[w] += m.at(h, w) * v.vector[h];
  return out;
        }
}

template <size_t S>
inline Vector<S> &softmax(const Vector<S> &v, Vector<S> &out) {
  float max = v.vector[0];
  float sum = 0;

  for (size_t i = 1; i < S; ++i)
    if (max < v.vector[i]) max = v.vector[i];

  for (size_t i = 0; i < S; i++) sum += std::exp(v.vector[i] - max);

  float c = std::max(sum, 10e-8f);

  for (size_t i = 0; i < S; i++)
    out.vector[i] = std::exp(v.vector[i] - max) / c;

  return out;
}

#endif  // NEUWURONKA_MATRIX_HPP
