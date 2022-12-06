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

#include "../struct.hpp"

namespace nn
{
    template <size_t _in_features, size_t _out_features, bool _input = false>
    // Allows us to define network architecture
    struct Linear
    {
        static constexpr size_t in_features = _in_features;
        static constexpr size_t out_features = _out_features;
        static constexpr bool input = _input;

        static constexpr const auto &activation_function = relu<float>;
        static constexpr const auto &activation_function_prime = relu_prime<float>;
    };

    template <typename... args>
    class MLP;

    template <typename module_t, typename... args>
    struct MLP<module_t, args...>
    {
        using network_t = MLP<args...>;
        using input_t = Vector<module_t::in_features>;
        using output_t = Vector<module_t::out_features>;
        using weights_t = Matrix<module_t::out_features, module_t::in_features>;
        using bias_t = output_t;

        network_t network;

        output_t weighted_input;
        output_t activation;
        output_t activation_prime;

        weights_t weights;
        bias_t bias;

        weights_t grad_w;
        bias_t grad_b;

        weights_t momentum_w;
        bias_t momentum_b;

        weights_t delta_w;
        bias_t delta_b;

        inline void zero_grad()
        {
            grad_b.zero();
            grad_w.zero();
            network.zero_grad();
        }

        inline void update(float learning_rate, float momentum)
        {
            // update friction
            momentum_w *= momentum;
            momentum_b *= momentum;

            // add new gradient
            momentum_w += grad_w * learning_rate;
            momentum_b += grad_b * learning_rate;

            // update parameters
            weights -= momentum_w;
            bias -= momentum_b;

            // recursively update the rest of network
            network.update(learning_rate, momentum);
        }

        template <typename predict_t>
        void update_mini_batch(std::vector<std::tuple<input_t, predict_t>> &data_and_labels, size_t start, size_t end,
                               float learning_rate, float momentum)
        {
            if constexpr (module_t::input)
            {
                auto n = static_cast<float>(end - start);

                // another catch from https://twitter.com/karpathy/status/1013244313327681536 :)
                zero_grad();

                static input_t store;

                for (size_t i = start; i < end; ++i)
                {
                    // set activations
                    forward(std::get<0>(data_and_labels[i]));
                    // set gradients
                    backward(std::get<0>(data_and_labels[i]), std::get<1>(data_and_labels[i]), store);
                }

                update(learning_rate / n, momentum);
            }
        }

        template <size_t NUM_SAMPLES, size_t EPOCHS, size_t BATCH_SIZE, typename predict_t>
        void SGD(std::vector<std::tuple<input_t, predict_t>> &data_and_labels, float learning_rate, float momentum,
                 float decay)
        {
            if (module_t::input)
            {
                std::mt19937 gen(42); // NOLINT
                auto lr = learning_rate;

                // in each epoch
                for (size_t e = 0; e < EPOCHS; ++e)
                {
                    std::cout << "Epoch " << e + 1 << "/" << EPOCHS << "\n";

                    // update learning rate wrt. decay
                    lr *= (1.0f / (1.0f + decay * static_cast<float>(e)));

                    // shuffle training set
                    std::shuffle(data_and_labels.begin(), data_and_labels.end(), gen);

                    // for each minibatch
                    for (size_t batch_index = 0; batch_index < NUM_SAMPLES; batch_index += BATCH_SIZE)

                        // perform one step of SGD wrt. momentum
                        update_mini_batch(data_and_labels, batch_index, std::min(NUM_SAMPLES, batch_index + BATCH_SIZE), lr,
                                          momentum);
                }
            }
        }

        template <typename predict_t>
        inline input_t &backward(const input_t &input, const predict_t &predict, input_t &store)
        {
            // gradient from layer above
            network.backward(activation, predict, delta_b);

            // derivative of activation function
            delta_b *= activation_prime;
            
            // compute my gradient
            dot_vector_transposed_vector(delta_b, input, delta_w);
            grad_b += delta_b;
            grad_w += delta_w;

            // return my gradient to layer below
            return dot_matrix_transposed_vector(weights, delta_b, store);
        }

        inline const auto &forward(const input_t &input)
        {
            // perform linear transformation
            dot_matrix_vector_transposed(weights, input, weighted_input) + bias;

            // set activation
            map(module_t::activation_function, weighted_input, activation);

            // set derivative of activation, will be used during backprop
            map(module_t::activation_function_prime, weighted_input, activation_prime);

            // pass activation to next layer
            return network.forward(activation);
        }

        explicit MLP(std::mt19937 &gen)
            : network(gen),
              weighted_input(),
              activation(),
              activation_prime(),
              weights(),
              bias(),
              grad_w(),
              grad_b(),
              delta_w(),
              delta_b(),
              momentum_w(),
              momentum_b()
        {
            float lower = -(1.0 / std::sqrt(module_t::in_features));
            float upper = (1.0 / std::sqrt(module_t::in_features));
            auto distribution = std::uniform_real_distribution<float>(lower, upper);
            for (size_t h = 0; h < weights.height; ++h)
            {
                for (size_t w = 0; w < weights.width; ++w)
                    weights.at(h, w) = distribution(gen);
                bias[h] = distribution(gen);
            }
        }

        template <size_t NUM_SAMPLES, size_t EPOCHS, size_t BATCH_SIZE, typename predict_t>
        void fit(std::vector<std::tuple<input_t, predict_t>> &data_and_labels, float learning_rate,
                 float momentum, float decay)
        {
            SGD<NUM_SAMPLES, EPOCHS, BATCH_SIZE>(data_and_labels, learning_rate, momentum, decay);
        }

        auto predict(const input_t &v) { return forward(v).imax(); }

        template <typename input_t, typename predict_t>
        auto &predict(const std::vector<input_t> &data, std::vector<predict_t> &out)
        {
            for (const input_t &input : data)
                out.push_back(predict(input));
            return out;
        }
    };

    template <>
    struct MLP<>
    {
        explicit MLP(std::mt19937 &) {}

        template <typename predict_t>
        inline const predict_t &forward(const predict_t &input)
        {
            static predict_t activation;
            return softmax(input, activation);
        }

        template <typename predict_t>
        inline predict_t &backward(const predict_t &z, const predict_t &y, predict_t &store)
        {
            return cross_entropy_cost_function_prime(z, y, store);
        }

        inline void zero_grad() {}
        inline void update(float, float) {}
    };
}

template <size_t S>
inline Vector<S> &softmax(const Vector<S> &v, Vector<S> &out)
{
    float max = v.vector[0];
    float sum = 0;

    for (size_t i = 1; i < S; ++i)
        max = max < v[i] ? v[i] : max;

    for (size_t i = 0; i < S; i++)
        sum += std::exp(v[i] - max);

    float c = std::max(sum, 10e-8f);

    for (size_t i = 0; i < S; i++)
        out.vector[i] = std::exp(v[i] - max) / c;

    return out;
}

template <size_t S>
inline auto &cross_entropy_cost_function_prime(const Vector<S> &activation, const Vector<S> &y, Vector<S> &out)
{
    for (size_t i = 0; i < S; ++i)
        out[i] = activation[i] - y[i];
    return out;
}

#endif // NEUWURONKA_NETWORK_HPP