//
// Created by Martin Krebs on 19/10/2022.
//

#ifndef NEUWURONKA_NETWORK_HPP
#define NEUWURONKA_NETWORK_HPP

#include <array>
#include <random>
#include <iostream>
#include "math.hpp"


// Modelling our network as a recursive structure allows us (surprisingly) to use recursive templates
template<typename ...args>
class Network {};


template<typename previous_layer, typename current_layer, typename  ...args>
class Network<previous_layer, current_layer, args...> {
    using input_t = Vector<previous_layer::size>;
    using output_t = Vector<current_layer::size>;
    using weights_t = Matrix<previous_layer::size, current_layer::size + 1>;
    using network_t = Network<current_layer, args...>;

    weights_t weights; // note this also includes the bias as the last element on each row
    weights_t gradient; // preallocate everything

    output_t output; //

    network_t network;

    void train_sgd() {
        // compute_gradient(gradient)
        // update_weights(gradient)
    }

    template<typename D, typename G>
    void constexpr init_weights(D &distribution, G &generator) {
        for (size_t h = 0; h < weights.height; ++h) {
            for (size_t w = 0; w < weights.width; ++w) {
                weights.at(h, w) = distribution(generator);
            }
        }
    }

    inline void update_weights(const weights_t &Q, float eta, int batch_size) {
        weights -= Q * (eta / static_cast<float>(batch_size));
    }

public:
    using predict_t = typename network_t::predict_t;

    predict_t feedforward(const input_t &input) {
        std::cout << "My input is:  " << input.to_string() << "\n"; // FIXME

        // transform using weights
        weights.weight_bias_dot(input, output);

        std::cout << "My weights are:\n" << weights.to_string() << "\n"; // FIXME

        // apply activation function
        output.map(current_layer::activation_function);

        std::cout << "My output is: " << output.to_string() << "\n"; // FIXME


        return network.feedforward(output);
    }

    output_t backprop(const input_t& activation, const predict_t &ground_truth) {
        auto z = weights.weight_bias_dot(activation);
        if constexpr (current_layer::output) {
            return z.map(current_layer::activation_function) - ground_truth;
        } else {
            auto delta = activation * z.map(current_layer::activation_function_prime);;
        }
    }

    Network() {
        // TODO: this needs not to be instantiated pre layer
        std::random_device rd;
        std::mt19937 gen(rd());

        if constexpr (current_layer::output) { // Xavier
            constexpr float lower = -(1.0 / ce_sqrt(previous_layer::size));
            constexpr float upper = (1.0 / ce_sqrt(previous_layer::size));
            auto distribution = std::uniform_real_distribution<float>(lower, upper);
            init_weights(distribution, gen);
        } else { // He
            auto distribution = std::normal_distribution<float>(0.0, ce_sqrt(2.0 / previous_layer::size));
            init_weights(distribution, gen);
        }
    }

    int predict(const input_t& v) {
        return feedforward(v).imax();
    }

    // TODO fit

};

template<typename output_layer>
class Network<output_layer> {
public:
    using predict_t = Vector<output_layer::size>;

    Network() = default;

    predict_t feedforward(const predict_t &input) {
        // TODO apply softmax
        std::cout << "Last predict";
        return input;
    }
};

#endif //NEUWURONKA_NETWORK_HPP
