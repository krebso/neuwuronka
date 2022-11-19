//
// Created by Martin Krebs on 12/11/2022.
//

#ifndef NEUWURONKA_MATH_HPP
#define NEUWURONKA_MATH_HPP

#include <cmath>
#include <limits>

#include "vector.hpp"

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
    float sum, c;

    for (size_t i = 1; i < S; ++i)
        if (max < v.vector[i]) max = v.vector[i];

    for (size_t i = 0; i < S; ++i) sum += std::exp(v[i] - max);

    c = max + std::log(sum);

    for (size_t i = 0; i < S; ++i) out.vector[i] = exp(v.vector[i] - c);

    return out;
}

template <size_t S>
inline auto &mse_cost_function_prime(const Vector<S> &activation, const Vector<S> &y, Vector<S> &out) {
    for (size_t i = 0; i < S; ++i) out.vector[i] = activation.vector[i] - y.vector[i];
    return out;
}

#endif  // NEUWURONKA_MATH_HPP
