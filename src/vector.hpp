//
// Created by Martin Krebs on 17/10/2022.
//

#ifndef NEUWURONKA_VECTOR_HPP
#define NEUWURONKA_VECTOR_HPP

template<size_t N>
class vector {
    std::array<double, N> _vector;

public:
    vector() = default;

    vector<N> operator+(vector<N> other) const {
        vector<N> sum;

        for (size_t i = 0; i < _vector.size(); i++)
            sum[i] = this[i] + other[i];

        return sum;
    }

    vector<N> operator+=(vector<N> other) const {
        for (size_t i = 0; i < _vector.size(); i++)
            this[i] += other[i];

        return this;
    }

    vector<N> operator-(vector<N> other) const {
        for (size_t i = 0; i < _vector.size(); i++)
            this[i] -= other[i];

        return this;
    }

    double dot(vector<N> other) const {
        double prod;

        for (size_t i = 0; i < _vector.size(); i++)
            prod += this[i] * other[i];

        return prod;
    }

    vector<N> operator*(double k) const {
        vector<N> prod;

        for (size_t i = 0; i < _vector.size(); i++)
            prod += this[i] * k;

        return prod;
    }

    vector<N>& operator*=(double k) {
        for (double& d : _vector)
            d *= k;

        return this;
    }

};

#endif //NEUWURONKA_VECTOR_HPP
