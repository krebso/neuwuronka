//
// Created by Martin Krebs on 12/11/2022.
//

#ifndef NEUWURONKA_MATH_HPP
#define NEUWURONKA_MATH_HPP

#include <cmath>
#include <limits>

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
