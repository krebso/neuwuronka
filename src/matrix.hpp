//
// Created by Martin Krebs on 15/10/2022.
//

#ifndef NEUWURONKA_MATRIX_HPP
#define NEUWURONKA_MATRIX_HPP

#include <vector>

template<size_t height, size_t width>
class matrix {
    std::array<std::array<float, width>, height> _matrix;

public:
    matrix() = default;
};

#endif //NEUWURONKA_MATRIX_HPP
