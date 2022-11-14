//
// Created by Martin Krebs on 13/11/2022.
//

#ifndef NEUWURONKA_LAYER_HPP
#define NEUWURONKA_LAYER_HPP

// Building block of the whole framework
// Each network is constructed using layers
// This is just an interface allowing me to recursively define the whole network at compile time using recursive templates

#include <random>
#include <iostream>
#include <string>

template<size_t NEURONS, typename N = float>
struct Layer {
    static constexpr size_t size = NEURONS;
    static constexpr bool input = false;
    static constexpr bool output = false;

    // static constexpr const auto& activation_function = relu<N>;
    // static constexpr const auto& activation_function_prime = relu_prime<N>;

    static constexpr const auto& activation_function = id<N>;
    static constexpr const auto& activation_function_prime = id<N>;
};

template<size_t NEURONS, typename N = float>
struct HiddenLayer : Layer<NEURONS> {};


template<size_t NEURONS, typename N = float>
struct InputLayer : Layer<NEURONS> {
    static constexpr bool input = true;
};


template<size_t NEURONS, typename N = float>
struct OutputLayer : Layer<NEURONS> {
    static constexpr bool output = true;

    static constexpr const auto& activation_function = id<N>;
    static constexpr const auto& activation_function_prime = id<N>;

    // static constexpr const auto& activation_function = sigmoid<N>;
    // static constexpr const auto& activation_function_prime = sigmoid_prime<N>;
};

#endif //NEUWURONKA_LAYER_HPP
