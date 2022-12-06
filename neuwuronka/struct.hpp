//
// Created by Martin Krebs on 15/10/2022.
//

#ifndef NEUWURONKA_STRUCT_HPP
#define NEUWURONKA_STRUCT_HPP

#include <vector>

#include "math.hpp"

/*
 Important, all the operators are not const! Because we want to avoid creating copies, v1 + v2 will modify v1!
*/

template <size_t N>
struct Vector {
    static constexpr size_t size = N;
    std::vector<float> vector;

    Vector()
        :  vector(N) {}

    void zero() {
        for (auto& v : vector) v = 0.0f;
    }

    float operator[](size_t i) const { return vector[i]; }

    float& operator[](size_t i) { return vector[i]; }

    Vector<N>& operator+(const Vector<N>& other) {
#pragma clang loop vectorize(assume_safety)
#pragma clang loop unroll_count(2)
        for (size_t i = 0; i < N; i++) vector[i] += other[i];
        return *this;
    }

    Vector<N>& operator+=(const Vector<N>& other) {
#pragma clang loop vectorize(assume_safety)
        for (size_t i = 0; i < N; i++) vector[i] += other[i];
        return *this;
    }

    Vector<N>& operator-(Vector<N> other) {
#pragma clang loop vectorize(assume_safety)
        for (size_t i = 0; i < N; i++) vector[i] -= other[i];
        return *this;
    }

    Vector<N>& operator-=(Vector<N> other) {
#pragma clang loop vectorize(assume_safety)
        for (size_t i = 0; i < N; i++) vector[i] -= other[i];
        return *this;
    }

    Vector<N>& operator*(Vector<N> other) {
#pragma clang loop vectorize(assume_safety)
        for (size_t i = 0; i < N; i++) vector[i] *= other[i];
        return *this;
    }

    Vector<N>& operator*=(Vector<N> other) {
#pragma clang loop vectorize(assume_safety)
        for (size_t i = 0; i < N; i++) vector[i] *= other[i];
        return *this;
    }

    Vector<N>& operator*=(float k) {
#pragma clang loop vectorize(assume_safety)
        for (size_t i = 0; i < N; ++i) vector[i] *= k;
        return *this;
    }

    Vector<N>& operator*(float k) {
#pragma clang loop vectorize(assume_safety)
        for (size_t i = 0; i < N; ++i) vector[i] *= k;
        return *this;
    }

    [[nodiscard]] size_t imax() const {
        size_t i = 0;
        for (size_t j = 1; j < size; ++j)
            if (vector[j] > vector[i]) i = j;
        return i;
    }
};
template <size_t H, size_t W>
struct Matrix {
    static constexpr size_t height = H;
    static constexpr size_t width = W;
    std::vector<float> matrix;

    Matrix()
        : matrix(H * W) {}

    void zero() { for (auto &v : matrix) v = 0; }

    constexpr float &at(size_t h, size_t w) { return matrix[h * W + w]; }

    [[nodiscard]] constexpr const float &at(size_t h, size_t w) const { return matrix[h * W + w]; }

    inline Matrix<H, W> &operator+=(const Matrix<H, W> &other) {
#pragma clang loop vectorize(assume_safety)
        for (size_t i = 0; i < H; i++)
#pragma clang loop unroll_count(2)
            for (size_t j = 0; j < W; j++) at(i, j) += other.at(i, j);
        return *this;
    };

    inline Matrix<H, W> &operator-=(const Matrix<H, W> &other) {
#pragma clang loop vectorize(assume_safety)
        for (size_t i = 0; i < H; i++)
#pragma clang loop unroll_count(2)
            for (size_t j = 0; j < W; j++) at(i, j) -= other.at(i, j);
        return *this;
    };

    inline auto &operator*(float k) {
#pragma clang loop vectorize(assume_safety)
        for (size_t i = 0; i < H * W; ++i) matrix[i] *= k;
        return *this;
    }

    inline auto &operator*=(float k) {
#pragma clang loop vectorize(assume_safety)
        for (size_t i = 0; i < H * W; ++i) matrix[i] *= k;
        return *this;
    }
};

template <typename F, typename V>
auto &map(const F& f, const V& v, V& out) {
#pragma clang loop vectorize(assume_safety)
    for (size_t i = 0; i < V::size; ++i) out[i] = f(v[i]);
    return v;
}

template <size_t H, size_t W>
auto &dot_matrix_vector_transposed(const Matrix<H, W> &m, const Vector<W> &v, Vector<H> &out) {
    out.zero();
#pragma clang loop vectorize(assume_safety)
    for (size_t h = 0; h < H; ++h)
#pragma clang loop unroll_count(2)
        for (size_t w = 0; w < W; ++w) out[h] += m.at(h, w) * v[w];

    return out;
}

template <size_t H, size_t W>
auto &dot_vector_transposed_vector(const Vector<H> &hv, const Vector<W> &wv, Matrix<H, W> &out) {
#pragma clang loop vectorize(assume_safety)
    for (size_t h = 0; h < H; ++h)
#pragma clang loop unroll_count(2)
        for (size_t w = 0; w < W; ++w) out.at(h, w) = hv[h] * wv[w];
    return out;
}

template <size_t H, size_t W>
auto &dot_matrix_transposed_vector(const Matrix<H, W> &m, const Vector<H> &v, Vector<W> &out) {
    out.zero();
#pragma clang loop vectorize(assume_safety)
    for (size_t h = 0; h < H; ++h)
#pragma clang loop unroll_count(2)
        for (size_t w = 0; w < W; ++w) out[w] += m.at(h, w) * v[h];
    return out;
}

#endif  // NEUWURONKA_STRUCT_HPP
