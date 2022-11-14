//
// Created by Martin Krebs on 12/11/2022.
//

#ifndef NEUWURONKA_MATH_HPP
#define NEUWURONKA_MATH_HPP

#include <cmath>
#include <limits>

template<typename N>
inline N id(N n) {
    return n;
}

template<typename N>
inline N id_prime(N n) {
    return 1;
}

template<typename N>
inline N relu(N n) {
    return std::max(n, 0.0f);
}

template<typename N>
inline N relu_prime(N n) {
    return static_cast<bool>(n > 0);
}

template<typename N>
inline N sigmoid(N n) {
    return 1 / (1 + std::exp(-n));
}

template<typename N>
inline N sigmoid_prime(N n) {
    return sigmoid(n) * (1 - sigmoid(n));
}

template<size_t S>
inline Vector<S>& softmax(Vector<S> &v) {
    float sum = 0;

    for (size_t i = 0; i < S; ++i)
        sum += v[i];

    for (size_t i = 0; i < S; ++i)
        v[i] /= sum;

    return v;
}

template<size_t S>
inline Vector<S> cross_entropy_loss(const Vector<S> &onehot, const Vector<S> &predicted) { // TODO: add output vector here
    // TODO: Make SURE the predicted is after softmax, otherwise it will blow up
    Vector<S> loss;

    for (size_t i = 0; i < S; ++i)
        loss[i] = -onehot[i] * log(predicted[i]); // TODO: optimize this

    return loss;
}

// https://gist.github.com/alexshtf/eb5128b3e3e143187794, found while looking for native constexpr sqrt
#pragma clang diagnostic push
#pragma ide diagnostic ignored "misc-no-recursion"
float constexpr sqrt_newton_raphson(float x, float curr, float prev) {
    return curr == prev
           ? curr
           : sqrt_newton_raphson(x, 0.5f * (curr + x / curr), curr);
}
#pragma clang diagnostic pop


float constexpr ce_sqrt(float x) noexcept {
    return x >= 0 && x < std::numeric_limits<float>::infinity()
           ? sqrt_newton_raphson(x, x, 0)
           : std::numeric_limits<float>::quiet_NaN();
}

#endif //NEUWURONKA_MATH_HPP
