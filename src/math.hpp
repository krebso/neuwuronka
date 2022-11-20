//
// Created by Martin Krebs on 12/11/2022.
//

#ifndef NEUWURONKA_MATH_HPP
#define NEUWURONKA_MATH_HPP

#include <cmath>
#include <limits>

#include "vector.hpp"

template <typename N>
inline N one_zero(N n) {
    return n > 0.0f ? 1.0f : 0.0f;
}

template <typename N>
inline N relu(N n) {
    return std::max(n, 0.0f);
}

template <typename N>
inline N relu_prime(N n) {
    return n > 0 ? 1.0f : 0.0f;
}

template <typename N>
inline N sigmoid(N n) {
    return 1.0f / (1.0f + std::exp(-n));
}

template <typename N>
inline N sigmoid_prime(N n) {
    return sigmoid(n) * (1.0f - sigmoid(n));
}

template <size_t S>
inline Vector<S> &softmax(const Vector<S> &v, Vector<S> &out) {
    float max = v.vector[0];
    float sum = 0;

    for (size_t i = 1; i < S; ++i)
        if (max < v.vector[i]) max = v.vector[i];

    for (size_t i = 0; i < S; i++) sum += std::exp(v.vector[i] - max);

    float c = std::max(sum, 10e-8f);

    for (size_t i = 0; i < S; i++) out.vector[i] = std::exp(v.vector[i] - max) / c;

    return out;
}

template <size_t S>
inline auto &cross_entropy_cost_function_prime(const Vector<S> &activation, const Vector<S> &y, Vector<S> &out) {
    for (size_t i = 0; i < S; ++i) out.vector[i] = activation.vector[i] - y.vector[i];
    return out;
}

template <size_t S>
inline auto &mse_cost_function_prime(const Vector<S> &activation, const Vector<S> &y, Vector<S> &out) {
    for (size_t i = 0; i < S; ++i) out.vector[i] = activation.vector[i] - y.vector[i];
    return out;
}
#endif  // NEUWURONKA_MATH_HPP
