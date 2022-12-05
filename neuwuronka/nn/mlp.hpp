//
// Created by Martin Krebs on 19/10/2022.
//

#ifndef NEUWURONKA_NETWORK_NEW_HPP
#define NEUWURONKA_NETWORK_NEW_HPP

#include <algorithm>
#include <array>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#include "module.hpp"

namespace nn
{

    template <typename... args>
    class MLP;

    template <typename module_t, typename... args>
    struct MLP<module_t, args...>
    {
        using network_t = MLP<args...>;
        using input_t = typename module_t::input_t;

        module_t module;
        network_t network;

        explicit MLP(std::mt19937 &generator)
            : network(generator),
              module(generator) {}

        inline const auto &forward(const input_t &z)
        {
            return network.forward(module.forward(z));
        }

        template <typename predict_t>
        inline const input_t &backward(const input_t &z, const predict_t &predict, input_t &store)
        {
            const auto &delta = network.backward(module.activation, predict, module.store);
            return module.backward(z, delta, store);
        }

        inline void zero_grad()
        {
            module.zero_grad();
            network.zero_grad();
        }

        inline void update(float lr, float momentum)
        {
            module.update(lr, momentum);
            network.update(lr, momentum);
        }

        template <typename predict_t>
        void update_mini_batch(
            std::vector<std::tuple<input_t, predict_t>> &data_and_labels,
            size_t start, size_t end, float learning_rate, float momentum)
        {
            static input_t store;
            auto n = static_cast<float>(end - start);

            zero_grad();

            for (size_t i = start; i < end; ++i)
            {
                forward(std::get<0>(data_and_labels[i]));
                backward(std::get<0>(data_and_labels[i]), std::get<1>(data_and_labels[i]), store);
            }

            update(learning_rate / n, momentum);
        }

        template <size_t NUM_SAMPLES, size_t EPOCHS, size_t BATCH_SIZE, typename predict_t>
        void SGD(std::vector<std::tuple<input_t, predict_t>> &data_and_labels,
                 float learning_rate, float momentum, float)
        {
            if constexpr (module_t::input)
            {
                std::mt19937 gen(42); // NOLINT
                auto lr = learning_rate;

                for (size_t e = 1; e < EPOCHS + 1; ++e)
                {
                    std::cout << "Epoch " << e << "/" << EPOCHS << "\n";
                    // lr *= (1.0f / (1.0f + decay * static_cast<float>(e)));

                    std::shuffle(data_and_labels.begin(), data_and_labels.end(), gen);

                    for (size_t batch_index = 0; batch_index < NUM_SAMPLES; batch_index += BATCH_SIZE)
                        update_mini_batch(data_and_labels, batch_index, std::min(NUM_SAMPLES, batch_index + BATCH_SIZE), lr, momentum);
                }
            }
        }

        template <size_t NUM_SAMPLES, size_t EPOCHS, size_t BATCH_SIZE, typename predict_t>
        void fit(std::vector<std::tuple<input_t, predict_t>> &data_and_labels,
                 float learning_rate, float momentum, float decay)
        {
            SGD<NUM_SAMPLES, EPOCHS, BATCH_SIZE>(data_and_labels, learning_rate,
                                                 momentum, decay);
        }

        auto predict(const input_t &input) { return forward(input).imax(); }

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
        explicit MLP(std::mt19937 &){};

        template <typename predict_t>
        inline const predict_t &forward(const predict_t &input)
        {
            static predict_t out;
            return softmax(input, out);
        }

        template <typename predict_t>
        inline const predict_t &backward(const predict_t &z, const predict_t &predict, predict_t &store)
        {
            for (size_t i = 0; i < predict_t::size; ++i)
                store[i] = z[i] - predict[i];
            return store;
        }

        inline void zero_grad() {}
        inline void update(float, float) {}
    };

} // namespace nn

#endif // NEUWURONKA_NETWORK_NEW_HPP
