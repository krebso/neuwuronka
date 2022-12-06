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

        weights_t delta_grad_w;
        bias_t delta_grad_b;

        inline void zero_grad()
        {
            grad_b.zero();
            grad_w.zero();
            network.zero_grad();
        }

        inline void update(float learning_rate, float momentum)
        {
            momentum_w *= momentum;
            momentum_b *= momentum;

            momentum_w += grad_w * learning_rate;
            momentum_b += grad_b * learning_rate;

            weights -= momentum_w;
            bias -= momentum_b;

            network.update(learning_rate, momentum);
        }

        template <typename predict_t>
        void update_mini_batch(std::vector<std::tuple<input_t, predict_t>> &data_and_labels, size_t start, size_t end,
                               float learning_rate, float momentum)
        {
            if constexpr (module_t::input)
            {
                auto n = static_cast<float>(end - start);

                zero_grad();

                static input_t store;

                for (size_t i = start; i < end; ++i)
                {
                    forward(std::get<0>(data_and_labels[i]));
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

                for (size_t e = 0; e < EPOCHS; ++e)
                {
                    std::cout << "Epoch " << e + 1 << "/" << EPOCHS << "\n";
                    lr *= (1.0f / (1.0f + decay * static_cast<float>(e)));
                    std::shuffle(data_and_labels.begin(), data_and_labels.end(), gen);

                    for (size_t batch_index = 0; batch_index < NUM_SAMPLES; batch_index += BATCH_SIZE)
                        update_mini_batch(data_and_labels, batch_index, std::min(NUM_SAMPLES, batch_index + BATCH_SIZE), lr,
                                          momentum);
                }
            }
        }

        template <typename predict_t>
        input_t &backward(const input_t &input, const predict_t &predict, input_t &store)
        {
            network.backward(activation, predict, delta_grad_b) * activation_prime;
            dot_vector_transposed_vector(delta_grad_b, input, delta_grad_w);
            grad_b += delta_grad_b;
            grad_w += delta_grad_w;
            return dot_matrix_transposed_vector(weights, delta_grad_b, store);
        }

        const auto &forward(const input_t &input)
        {
            dot_matrix_vector_transposed(weights, input, weighted_input) + bias;
            map(module_t::activation_function, weighted_input, activation);
            map(module_t::activation_function_prime, weighted_input, activation_prime);
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
              delta_grad_w(),
              delta_grad_b(),
              momentum_w(),
              momentum_b()
        {
            float lower = -(1.0 / std::sqrt(module_t::in_features));
            float upper = (1.0 / std::sqrt(module_t::in_features));
            auto distribution = std::uniform_real_distribution<float>(lower, upper);
            // auto distribution = std::normal_distribution<float>(0.0, std::sqrt(2.0 / module_t::in_features));
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
        if (max < v.vector[i])
            max = v.vector[i];

    for (size_t i = 0; i < S; i++)
        sum += std::exp(v.vector[i] - max);

    float c = std::max(sum, 10e-8f);

    for (size_t i = 0; i < S; i++)
        out.vector[i] = std::exp(v.vector[i] - max) / c;

    return out;
}

template <size_t S>
inline auto &cross_entropy_cost_function_prime(const Vector<S> &activation, const Vector<S> &y, Vector<S> &out)
{
    for (size_t i = 0; i < S; ++i)
        out.vector[i] = activation.vector[i] - y.vector[i];
    return out;
}

#endif // NEUWURONKA_NETWORK_HPP