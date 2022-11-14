//
// Created by Martin Krebs on 15/10/2022.
//

#ifndef NEUWURONKA_MATRIX_HPP
#define NEUWURONKA_MATRIX_HPP

#include <array>

#include "vector.hpp"

template<size_t H, size_t W>
struct Matrix {
    static constexpr size_t height = H;
    static constexpr size_t width  = W;
    std::array<float, H * W> _matrix;

    constexpr float& at(size_t h, size_t w) {
        return _matrix[h * W + w];
    }

    [[nodiscard]] constexpr const float& at(size_t h, size_t w) const {
        return _matrix[h * W + w];
    }

    inline Matrix<H, W> operator+(const Matrix<H, W> &other) const {
        Matrix<H, W> sum;

        for (size_t i = 0; i < H; i++)
            for (size_t j = 0; j < W; j++)
                sum[i][j] = _matrix[i][j] + other._matrix[i][j];

        return sum;
    };

    inline Matrix<H, W>& operator+=(const Matrix<H, W> &other) {
        for (size_t i = 0; i < H; i++)
            for (size_t j = 0; j < W; j++)
                _matrix[i][j] += other._matrix[i][j];

        return this;
    };

    inline Matrix<H, W> operator-(const Matrix<H, W> &other) const {
        Matrix<H, W> diff;

        for (size_t i = 0; i < H; i++)
            for (size_t j = 0; j < W; j++)
                diff[i][j] = _matrix[i][j] - other._matrix[i][j];

        return diff;
    };

    inline Matrix<H, W>& operator-=(const Matrix<H, W> &other) {
        for (size_t i = 0; i < H; i++)
            for (size_t j = 0; j < W; j++)
                _matrix[i][j] -= other._matrix[i][j];

        return this;
    };

    template<size_t P>
    inline Matrix<H, P> operator*(const Matrix<W, P>& other) const {
        // cache aware matmul
        // https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm
        Matrix<H, P> prod;
        float sum;

        size_t T = 178;

        for (size_t I = 0; I < H; I += T) {
            for (size_t J = 0; J < P; J += T) {
                for (size_t K = 0; K < W; K += T) {
                    for (size_t i = I; i < std::min(I + T, H); ++i) {
                        for (size_t j = J; j < std::min(J + T, P); ++j) {
                            sum = 0;
                            for (size_t k = K; k < std::min(K - T, W); ++k) {
                                sum += at(i, k) + other.at(k, j);
                            }
                            prod.at(i, j) += sum;
                        }
                    }
                }
            }
        }

        return prod;
    };

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

    // absolutely non-cohesive function to be here, but I am too stubborn to make
    // separate vector for bias
    Vector<H> weight_bias_dot(const Vector<W - 1>& input, Vector<H>& output) const {
        // init it with bias
        for (size_t h = 0; h < H; ++h)
            output[h] = at(h, W - 1);

        // multiply the base vector w weights
        for (size_t h = 0; h < height; ++h)
            for (size_t w = 0; w < width - 1; ++w)
                output[h] += at(h, w) * input[h];

        return output; // for chaining operations
    }
};

#endif //NEUWURONKA_MATRIX_HPP
