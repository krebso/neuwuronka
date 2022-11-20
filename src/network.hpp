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

    weights_t momentum_nabla_w;  // momentum weights history
    biases_t momentum_nabla_b;   // same as above for biases

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

    void update_weights(float learning_rate, float momentum) {
        // update the friction
        momentum_nabla_w * momentum;
        momentum_nabla_b * momentum;

        // update history with current nablas
        momentum_nabla_w += nabla_w * learning_rate;
        momentum_nabla_b += nabla_b * learning_rate;

        // update weights and biases
        weights -= momentum_nabla_w;
        biases -= momentum_nabla_b;

        // update the rest of the network
        if constexpr(!current_layer::output) network.update_weights(learning_rate, momentum);
    }

    void update_mini_batch(std::vector<std::tuple<input_t, predict_t>> &data_and_labels, size_t start, size_t end,
                           float learning_rate, float momentum) {
        // num of samples in minibatch
        auto n = static_cast<float>(end - start);

        // recursively zero out both nablas
        zero_nablas();

        // for each training sample
        for (size_t i = start; i < end; ++i) {
            auto x = std::get<0>(data_and_labels[i]);

            // set activations, (both actual and prime) and weighted_inputs
            feedforward(x);

            // compute the gradient using backpropagation
            backprop(std::get<0>(data_and_labels[i]), std::get<1>(data_and_labels[i]));
        }

        // update the weights and biases recursively for whole network
        update_weights(learning_rate / n, momentum);
    }

    template <size_t NUM_SAMPLES, size_t EPOCHS, size_t BATCH_SIZE>
    void SGD(std::vector<std::tuple<input_t, predict_t>> &data_and_labels, float learning_rate, float momentum, float decay) {
        if (previous_layer::input) {
            std::mt19937 gen(42);  // NOLINT

            // in each epoch
            for (size_t e = 0; e < EPOCHS; ++e) {

                // compute the learning rate for current epoch wrt to decay
                // auto lr = learning_rate * (1.0f / (1.0f + decay * static_cast<float>(e)));
                auto lr = learning_rate;

                std::cout << "Epoch " << e + 1 << "/" << EPOCHS << "\n";

                // shuffle the data randomly
                std::shuffle(data_and_labels.begin(), data_and_labels.end(), gen);

                // for all mini-batches
                for (size_t batch_index = 0; batch_index < NUM_SAMPLES; batch_index += BATCH_SIZE)
                    // perform gradient descent on single minibatch
                    update_mini_batch(data_and_labels, batch_index, std::min(NUM_SAMPLES, batch_index + BATCH_SIZE),
                                      lr, momentum);
            }
        }
    }

    const biases_t &backprop(const input_t &input, const predict_t &predict) {
        if constexpr (current_layer::output) {
            cross_entropy_cost_function_prime(activation, predict, delta_nabla_b);
        } else {
            // get the delta from next layer
            auto &next_delta_nabla_b = network.backprop(activation, predict);

            // multiply the corresponding matrix
            dot_matrix_transposed_vector(network.weights, next_delta_nabla_b, delta_nabla_b);
        }

        delta_nabla_b *= activation_prime;

        if constexpr (current_layer::output) {
            // std::cout << "[Output layer] Delta nabla b: " << delta_nabla_b.to_string() << "\n";
        } else {
            // std::cout << "[Hidden layer] Delta nabla b: " << delta_nabla_b.to_string() << "\n";
        }

        dot_vector_transposed_vector(delta_nabla_b, input, delta_nabla_w);

        // std::cout << "Delta nabla w: \n" << delta_nabla_w.to_string() << "\n";

        nabla_b += delta_nabla_b;
        nabla_w += delta_nabla_w;

        // std::cout << "Leaving backprop\n";

        return delta_nabla_b;
    }

    predict_t feedforward(const input_t &input) {
        // // std::cout << "Feedforward\n";

        // compute weighted input
        dot_matrix_vector_transposed(weights, input, weighted_input);
        weighted_input += biases;

        // // std::cout << "Weighted input: " << weighted_input.to_string() << "\n";

        // apply activation function
        map(current_layer::activation_function, weighted_input, activation);

        // // std::cout << "Activation: " << activation.to_string() << "\n";

        // pre-compute first derivative of activation function, will be used during backprop
        map(current_layer::activation_function_prime, weighted_input, activation_prime);

        // // std::cout << "Activation prime: " << activation_prime.to_string() << "\n";

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
          delta_nabla_b(),
          momentum_nabla_w(),
          momentum_nabla_b() {
        if constexpr (true || current_layer::output) {  // Xavier
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
    void fit(std::vector<std::tuple<input_t, predict_t>> &data_and_labels, float learning_rate = 0.1f, float momentum = 0.9f, float decay = 0.0f) {
        SGD<NUM_SAMPLES, EPOCHS, BATCH_SIZE>(data_and_labels, learning_rate, momentum, decay);
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

    predict_t feedforward(const predict_t &input) {
        return softmax(input, activation);
    }

    void zero_nablas(){};
};

#endif  // NEUWURONKA_NETWORK_HPP
