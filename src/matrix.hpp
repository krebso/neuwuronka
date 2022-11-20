//
// Created by Martin Krebs on 15/10/2022.
//

#ifndef NEUWURONKA_MATRIX_HPP
#define NEUWURONKA_MATRIX_HPP

#include <array>

#include "math.hpp"
#include "vector.hpp"

template <size_t H, size_t W>
struct Matrix {
    static constexpr size_t height = H;
    static constexpr size_t width = W;
    std::array<float, H * W> matrix;

    void zero() { for (auto &v : matrix) v = 0; }

    constexpr float &at(size_t h, size_t w) { return matrix[h * W + w]; }

    [[nodiscard]] constexpr const float &at(size_t h, size_t w) const { return matrix[h * W + w]; }

    inline Matrix<H, W> &operator+=(const Matrix<H, W> &other) {
        #pragma clang loop vectorize(assume_safety)
        for (size_t i = 0; i < H; i++)
            #pragma clang loop unroll_count(10)
            for (size_t j = 0; j < W; j++) at(i, j) += other.at(i, j);
        return *this;
    };

    inline Matrix<H, W> &operator-=(const Matrix<H, W> &other) {
        #pragma clang loop vectorize(assume_safety)
        for (size_t i = 0; i < H; i++)
            #pragma clang loop unroll_count(10)
            for (size_t j = 0; j < W; j++) at(i, j) -= other.at(i, j);
        return *this;
    };

    inline auto &operator*(float k) {
        #pragma clang loop vectorize(assume_safety)
        for (auto &v : matrix) v *= k;
        return *this;
    }
};

template <size_t H, size_t W>
auto &dot_matrix_vector_transposed(const Matrix<H, W> &m, const Vector<W> &v, Vector<H> &out) {
    out.zero();
    #pragma clang loop vectorize(assume_safety)
    for (size_t h = 0; h < H; ++h)
        #pragma clang loop unroll_count(10)
        for (size_t w = 0; w < W; ++w) out.vector[h] += m.at(h, w) * v[w];

    return out;
}

template <size_t H, size_t W>
auto &dot_vector_transposed_vector(const Vector<H> &hv, const Vector<W> &wv, Matrix<H, W> &out) {
    #pragma clang loop vectorize(assume_safety)
    for (size_t h = 0; h < H; ++h)
        #pragma clang loop unroll_count(10)
        for (size_t w = 0; w < W; ++w) out.at(h, w) = hv[h] * wv[w];
    return out;
}

template <size_t H, size_t W>
auto &dot_matrix_transposed_vector(const Matrix<H, W> &m, const Vector<H> &v, Vector<W> &out) {
    out.zero();
    #pragma clang loop vectorize(assume_safety)
    for (size_t h = 0; h < H; ++h)
        #pragma clang loop unroll_count(10)
        for (size_t w = 0; w < W; ++w) out.vector[w] += m.at(h, w) * v.vector[h];
    return out;
}

#endif  // NEUWURONKA_MATRIX_HPP
