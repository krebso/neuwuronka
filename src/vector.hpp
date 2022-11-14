//
// Created by Martin Krebs on 17/10/2022.
//

#ifndef NEUWURONKA_VECTOR_HPP
#define NEUWURONKA_VECTOR_HPP

#include <array>

template<size_t N, typename data_t = float>
struct Vector {
    static constexpr size_t size = N;
    std::array<data_t, N> vector;

public:
    Vector() = default;

    data_t operator[](size_t i) const {
        return vector[i];
    }

    data_t & operator[](size_t i) {
        return vector[i];
    }

    Vector<N> operator+(Vector<N> other) const {
        Vector<N> sum;

        for (size_t i = 0; i < vector.size(); i++)
            sum[i] = this[i] + other[i];

        return sum;
    }

    Vector<N> operator+=(Vector<N> other) const {
        for (size_t i = 0; i < vector.size(); i++)
            this[i] += other[i];

        return this;
    }

    Vector<N> operator-(Vector<N> other) const {
        for (size_t i = 0; i < vector.size(); i++)
            this[i] -= other[i];

        return this;
    }

    data_t operator*(Vector<N> other) const {
        data_t prod;

        for (size_t i = 0; i < vector.size(); i++)
            prod += this[i] * other[i];

        return prod;
    }

    Vector<N> k_multiply(data_t k) const {
        Vector<N> prod;

        for (size_t i = 0; i < vector.size(); i++)
            prod += this[i] * k;

        return prod;
    }

    Vector<N>& operator*=(data_t k) {
        for (double& d : vector)
            d *= k;

        return this;
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
            if (vector[j] > vector[i])
                i = j;
        return i;
    }

    template<typename F>
    auto map(F f) {
        for (size_t i = 0; i < size; ++i)
            vector[i] = f(vector[i]);
        return this;
    }
};

template<size_t S>
constexpr Vector<S> onehot(size_t i) noexcept {
    Vector<S> _onehot;
    _onehot[i] = 1.0;
    return _onehot;
}


#endif //NEUWURONKA_VECTOR_HPP
