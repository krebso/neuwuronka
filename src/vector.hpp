//
// Created by Martin Krebs on 17/10/2022.
//

#ifndef NEUWURONKA_VECTOR_HPP
#define NEUWURONKA_VECTOR_HPP

#include <array>

template <size_t N, typename data_t = float>
struct Vector {
    static constexpr size_t size = N;
    std::array<data_t, N> vector;

    Vector() = default;

    void zero() {
        for (auto& v : vector) v = 0;
    }

    data_t operator[](size_t i) const { return vector[i]; }

    data_t& operator[](size_t i) { return vector[i]; }

    Vector<N>& operator+(const Vector<N>& other) {
        for (size_t i = 0; i < N; i++) vector[i] += other[i];
        return *this;
    }

    Vector<N>& operator+=(const Vector<N>& other) {
        for (size_t i = 0; i < N; i++) vector[i] += other.vector[i];
        return *this;
    }

    Vector<N> operator-(Vector<N> other) {
        for (size_t i = 0; i < N; i++) vector[i] -= other.vector[i];
        return *this;
    }

    Vector<N> operator-=(Vector<N> other) {
        for (size_t i = 0; i < N; i++) vector[i] -= other.vector[i];
        return *this;
    }

    Vector<N> operator*(Vector<N> other) {
        for (size_t i = 0; i < N; i++) vector[i] *= other.vector[i];
        return *this;
    }

    Vector<N> operator*=(Vector<N> other) {
        for (size_t i = 0; i < N; i++) vector[i] *= other.vector[i];
        return *this;
    }

    Vector<N>& operator*=(data_t k) {
        for (double& d : vector) d *= k;

        return *this;
    }

    Vector<N> operator*(float k) {
        for (auto& v : vector) v *= k;
        return *this;
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

    [[nodiscard]] size_t imax() const {
        size_t i = 0;
        for (size_t j = 1; j < size; ++j)
            if (vector[j] > vector[i]) i = j;
        return i;
    }
};

template <typename F, typename V>
auto map(const F& f, const V& v, V& out) {
    for (size_t i = 0; i < V::size; ++i) out.vector[i] = f(v.vector[i]);
    return v;
}

template <size_t S>
constexpr Vector<S> onehot(size_t i) noexcept {
    Vector<S> _onehot;
    _onehot[i] = 1.0;
    return _onehot;
}

#endif  // NEUWURONKA_VECTOR_HPP