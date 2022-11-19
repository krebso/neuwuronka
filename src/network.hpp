//
// Created by Martin Krebs on 19/10/2022.
//

#ifndef NEUWURONKA_NETWORK_HPP
#define NEUWURONKA_NETWORK_HPP

#include <algorithm>
#include <array>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#include "math.hpp"
#include "matrix.hpp"

template <typename... args>
class Network;

template <typename previous_layer, typename current_layer, typename... args>
struct Network<previous_layer, current_layer, args...> {
    using network_t = Network<current_layer, args...>;
    using input_t = Vector<previous_layer::size>;
    using output_t = Vector<current_layer::size>;
    using predict_t = typename network_t::predict_t;
    using weights_t = Matrix<current_layer::size, previous_layer::size>;
    using biases_t = output_t;

    network_t network;  // rest of the recursively defined network

    output_t weighted_input;  // allows us to store inner potential vectors which will be
                              // used during backprop
    output_t activation;      // vector which is being passed to the next layer as
                              // an input, also used during backprop

    output_t activation_prime;  // output vector for computing first derivative of activation function

    weights_t weights;  // matrix of weights
    biases_t biases;    // vector of biases

    weights_t nabla_w;  // this is our aggregator for storing the deltas through
                        // one iteration of SGD
    biases_t nabla_b;   // same as above, but for biases

    weights_t delta_nabla_w;  // pre-allocated output parameter, used during
                              // back-prop for storing single delta
    biases_t delta_nabla_b;   // same as above

    template <typename D, typename G>
    void init_weights_and_biases(D &distribution, G &generator) {
        // given desired initial weight distribution, init weights and biases
        for (size_t h = 0; h < weights.height; ++h) {
            for (size_t w = 0; w < weights.width; ++w) weights.at(h, w) = distribution(generator);
            biases[h] = distribution(generator);
        }
    }

    void zero_nablas() {
        nabla_b.zero();
        nabla_w.zero();
        network.zero_nablas();
    }

    void update_weights(float eta) {
        // update weights as biases
        weights -= nabla_w * eta;
        biases -= nabla_b * eta;

        // update the rest of the network
        network.update_weights(eta);
    }

    void update_mini_batch(std::vector<std::tuple<input_t, predict_t>> &data_and_labels, size_t start, size_t end,
                           float learning_rate) {
        // num of samples in minibatch
        auto n = static_cast<float>(end - start);

        // recursively zero out both nablas
        zero_nablas();

        // for each training sample
        for (size_t i = start; i < end; ++i) {
            auto x = std::get<0>(data_and_labels[i]);

            // set activations and weighted_inputs
            feedforward(x);

            // compute the gradient using backpropagation
            backprop(std::get<0>(data_and_labels[i]), std::get<1>(data_and_labels[i]));
        }

        // update the weights and biases recursively for whole network
        update_weights(learning_rate / n);
    }

    template <size_t NUM_SAMPLES, size_t EPOCHS, size_t BATCH_SIZE>
    void SGD(std::vector<std::tuple<input_t, predict_t>> &data_and_labels) {
        if (previous_layer::input) {
            std::mt19937 gen(42);  // NOLINT

            float eta = 0.5;
            float decay = 0.000001;

            // in each epoch
            for (size_t e = 0; e < EPOCHS; ++e) {
                std::cout << "Epoch " << e + 1 << "/" << EPOCHS << "\n";
                std::cout << network.biases.to_string() << "\n";
                // compute the learning rate for current epoch wrt [momentum] and decay
                auto lr = eta * (1.0f / (1.0f + decay * static_cast<float>(e)));

                // shuffle the data randomly
                std::shuffle(data_and_labels.begin(), data_and_labels.end(), gen);

                // for all mini-batches
                for (size_t batch_index = 0; batch_index < NUM_SAMPLES; batch_index += BATCH_SIZE) {
                    // perform gradient descent on single minibatch
                    update_mini_batch(data_and_labels, batch_index, std::min(NUM_SAMPLES, batch_index + BATCH_SIZE),
                                      lr);
                }
            }
        }
    }

    const biases_t &backprop(const input_t &input, const predict_t &predict) {
        if constexpr (current_layer::output) {
            mse_cost_function_prime(activation, predict, delta_nabla_b);
        } else {
            // get the delta from next layer
            auto &next_delta_nabla_b = network.backprop(activation, predict);

            // multiply the corresponding matrix
            dot_matrix_transposed_vector(network.weights, next_delta_nabla_b, delta_nabla_b);

            // multiply with derivative of activation function
            delta_nabla_b *= activation_prime;
        }

        dot_vector_vector_transposed(delta_nabla_b, input, delta_nabla_w);

        nabla_b += delta_nabla_b;
        nabla_w += delta_nabla_w;

        return delta_nabla_b;
    }

    predict_t feedforward(const input_t &input) {
        // compute weighted input
        dot_matrix_vector_transposed(weights, input, weighted_input);
        weighted_input += biases;

        // apply activation function
        map(current_layer::activation_function, weighted_input, activation);

        // pre-compute first derivative of activation function, will be used during backprop
        map(current_layer::activation_function_prime, weighted_input, activation_prime);

        // pass the activation as an input to the next layer
        return network.feedforward(activation);
    }

    explicit Network(std::mt19937 &gen)
        : network(gen),
          weighted_input(),
          activation(),
          activation_prime(),
          weights(),
          biases(),
          nabla_w(),
          nabla_b(),
          delta_nabla_w(),
          delta_nabla_b() {
        if constexpr (current_layer::output) {  // Xavier
            float lower = -(1.0 / std::sqrt(previous_layer::size));
            float upper = (1.0 / std::sqrt(previous_layer::size));
            auto distribution = std::uniform_real_distribution<float>(lower, upper);
            init_weights_and_biases(distribution, gen);
        } else {  // He
            auto distribution = std::normal_distribution<float>(0.0, std::sqrt(2.0 / previous_layer::size));
            init_weights_and_biases(distribution, gen);
        }
    }

    template <size_t NUM_SAMPLES, size_t EPOCHS, size_t BATCH_SIZE>
    void fit(std::vector<std::tuple<input_t, predict_t>> &data_and_labels) {
        SGD<NUM_SAMPLES, EPOCHS, BATCH_SIZE>(data_and_labels);
    }

    auto predict(const input_t &v) { return feedforward(v).imax(); }

    template <typename input_t, typename predict_t>
    auto &predict(const std::vector<input_t> &data, std::vector<predict_t> &out) {
        for (const input_t &input : data) out.push_back(predict(input));
        return out;
    }
};

template <typename output_layer>
class Network<output_layer> {
   public:
    using predict_t = Vector<output_layer::size>;

    predict_t activation;

    explicit Network(std::mt19937 &) : activation(){};

    predict_t feedforward(const predict_t &input) { return softmax(input, activation); }
    // predict_t feedforward(const predict_t &input) { return input; }
    void update_weights(float lr){};
    void zero_nablas(){};
};

#endif  // NEUWURONKA_NETWORK_HPP
