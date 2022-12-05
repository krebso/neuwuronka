//
// Created by Martin Krebs on 04/12/2022.
//

#ifndef NEUWURONKA_MODULE_HPP
#define NEUWURONKA_MODULE_HPP

#include "../struct.hpp"

namespace nn
{
    struct Module
    {
        void zero_grad(){};
        void update(float, float){};
    };

    template <size_t S>
    struct ReLU : Module
    {
        static constexpr size_t in_features = S;
        static constexpr size_t out_features = S;
        using output_t = Vector<S>;
        using input_t = output_t;

        output_t activation;
        output_t activation_prime;
        output_t store;

        explicit ReLU(std::mt19937 &) : activation(), activation_prime() {}

        inline const output_t &forward(const input_t &input)
        {
            map(relu<float>, input, activation);
            map(relu_prime<float>, input, activation_prime);
            return activation;
        }

        inline const output_t &backward(const input_t &, const output_t &delta, input_t &store)
        {
            dot_vector_vector(delta, activation_prime, store);
            return store;
        }
    };

    template <size_t S>
    struct Softmax : Module
    {
        using input_t = Vector<S>;
        using output_t = Vector<S>;

        output_t activation;
        output_t activation_prime;
        output_t store;

        explicit Softmax(std::mt19937 &) : activation(), activation_prime() {}

        inline const output_t &forward(const input_t &input)
        {
            softmax(input, activation);
            return activation;
        }

        inline const input_t &backward(const input_t &z, const output_t &y, input_t &store)
        {
            for (size_t i = 0; i < S; ++i)
                store[i] = z[i] - y[i];
            return store;
        }
    };

    template <size_t _in_features, size_t _out_features, bool _input = false, bool _output = false>
    struct Layer
    {
        static constexpr size_t in_features = _in_features;
        static constexpr size_t out_features = _out_features;
        static constexpr bool input = _input;
        static constexpr bool output = _output;
        Layer() = default;
    };

    template <size_t in_features, size_t out_features, bool _input = false, bool _output = false>
    struct Linear : Layer<in_features, out_features, _input, _output>
    {
        using input_t = Vector<in_features>;
        using output_t = Vector<out_features>;
        using weights_t = Matrix<out_features, in_features>;
        using biases_t = output_t;

        weights_t weights;
        biases_t biases;

        output_t activation;
        output_t activation_prime;
        output_t store;

        weights_t grad_w;
        biases_t grad_b;

        weights_t delta_w;
        input_t delta_b;

        weights_t momentum_w;
        biases_t momentum_b;

        Linear() = default;

        explicit Linear(std::mt19937 &generator) : weights(),
                                                   biases(),
                                                   activation(),
                                                   grad_w(),
                                                   grad_b(),
                                                   activation_prime(),
                                                   momentum_b(),
                                                   momentum_w()
        {
            auto distribution = std::normal_distribution<float>(
                0.0, std::sqrt(2.0f / static_cast<float>(in_features)));
            for (size_t h = 0; h < weights.height; ++h)
            {
                for (size_t w = 0; w < weights.width; ++w)
                    weights.at(h, w) = distribution(generator);
            }
        };

        inline const output_t &forward(const input_t &input)
        {
            dot_matrix_vector_transposed(weights, input, activation) + biases;
            return activation;
        }

        inline const input_t &backward(const input_t &z, const output_t &delta, input_t &store)
        {
            dot_vector_transposed_vector(delta, z, delta_w);
            grad_b += delta;
            grad_w += delta_w;
            dot_matrix_transposed_vector(weights, activation_prime, store);
            return delta_b;
        }

        inline void zero_grad()
        {
            grad_b.zero();
            grad_w.zero();
        }

        inline void update(float lr, float momentum)
        {
            momentum_w *= momentum;
            momentum_b *= momentum;

            momentum_w += grad_w * lr;
            momentum_b += grad_b * lr;

            weights -= momentum_w;
            biases -= momentum_b;
        }
    };
}

#endif // NEUWURONKA_MODULE_HPP
