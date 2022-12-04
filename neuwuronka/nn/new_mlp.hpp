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

#include "../struct.hpp"

namespace new_nn {

  struct Module {
    void zero_grad() {};
    void update(float, float) {};
  };

  template<size_t S>
  struct ReLU : Module {
    static constexpr size_t in_features  = S;
    static constexpr size_t out_features = S;
    using output_t  = Vector<S>;
    using input_t   = output_t;

    output_t activation;
    output_t activation_prime;

    explicit ReLU(std::mt19937 &) :activation(), activation_prime() {}
    ReLU() = default;

    inline const output_t &forward(const input_t& input) {
        return map(relu_prime<float>, input, activation_prime), map(relu<float>, input, activation);
    }

    inline const output_t &backward(const input_t&, const output_t& grad) {
      return activation_prime * grad;
      }
  };

  template<size_t S>
  struct Softmax : Module {
      using input_t = Vector<S>;
      using output_t = Vector<S>;

      output_t activation;

      explicit Softmax(std::mt19937 &) : activation() {}

      inline const output_t &forward(const input_t &input) {
          return softmax(input, activation);
      }

      inline const input_t &backward(const input_t &input, const output_t &grad) {
          for (size_t i = 0; i < 1; ++i) activation[i] = input[i] - grad[i];
          return activation;
      }
  };

  template <size_t _in_features, size_t _out_features, bool _input = false, bool _output = false>
  struct Layer {
    static constexpr size_t in_features = _in_features;
    static constexpr size_t out_features = _out_features;
    static constexpr bool input = _input;
    static constexpr bool output = _output;
    Layer() = default;
  };

  template <size_t in_features, size_t out_features, bool _input = false, bool _output = false>
  struct Linear : Layer<in_features, out_features, _input, _output> {
    using input_t   = Vector<in_features>;
    using output_t  = Vector<out_features>;
    using weights_t = Matrix<out_features, in_features>;
    using biases_t  = output_t;

    weights_t weights;
    biases_t  biases;
    output_t  activation;

    weights_t grad_w;
    biases_t  grad_b;
    weights_t delta_grad_w;
    
    input_t   prime;

    weights_t momentum_w;
    biases_t  momentum_b;

    Linear() = default;

    explicit Linear(std::mt19937 &generator) :
    weights(),
    biases(),
    activation(),
    grad_w(),
    grad_b(),
    prime(),
    delta_grad_w(),
    momentum_b(),
    momentum_w() {
      auto distribution = std::normal_distribution<float>(
          0.0, std::sqrt(2.0f / static_cast<float>(in_features)));
      for (size_t h = 0; h < weights.height; ++h) {
        for (size_t w = 0; w < weights.width; ++w)
          weights.at(h, w) = distribution(generator);
        biases[h] = distribution(generator);
      }
    };

    inline const output_t &forward(const input_t& input) {
      return dot_matrix_vector_transposed(weights, input, activation) + biases;
    }

    inline const input_t &backward(const input_t &input, const output_t& grad) {
      dot_vector_transposed_vector(grad, input, delta_grad_w);
      grad_b += grad;
      grad_w += delta_grad_w;
      return dot_matrix_transposed_vector(weights, grad, prime);
    }

    inline void zero_grad() {
      grad_b.zero();
      grad_w.zero();
    }

    inline void update(float lr, float momentum) {
      momentum_w *= momentum;
      momentum_b *= momentum;

      momentum_w += grad_w * lr;
      momentum_b += grad_b * lr;

      weights -= momentum_w;
      biases  -= momentum_b;
    }
  };

  template <typename... args>
  class MLP;

  template <typename module_t, typename... args>
  struct MLP<module_t, args...> {
    using network_t = MLP<args...>;
    using input_t   = typename module_t::input_t;

    module_t  module;
    network_t network;
  
    explicit MLP(std::mt19937 &generator)
      : network(generator),
        module(generator) {}

    inline const auto &forward(const input_t &input) { return network.forward(module.forward(input)); }

    template <typename predict_t>
    inline const input_t &backward(const input_t &input, const predict_t &predict) {
      return module.backward(input, network.backward(module.activation, predict));
    }

    inline void zero_grad() {
      module.zero_grad();
      network.zero_grad();
    }

    inline void update(float lr, float momentum) {
      module.update(lr, momentum);
      network.update(lr, momentum);
    }
    
    template<typename predict_t>
    void update_mini_batch(
        std::vector<std::tuple<input_t, predict_t>> &data_and_labels,
        size_t start, size_t end, float learning_rate, float momentum) {
      auto n = static_cast<float>(end - start);

      zero_grad();

      for (size_t i = start; i < end; ++i) {
        auto &y    = std::get<0>(data_and_labels[i]);
        auto &pred = std::get<1>(data_and_labels[i]);

        forward(y);

        backward(y, pred);
      }

      update(learning_rate / n, momentum);
    }
  
    template <size_t NUM_SAMPLES, size_t EPOCHS, size_t BATCH_SIZE, typename predict_t>
    void SGD(std::vector<std::tuple<input_t, predict_t>> &data_and_labels,
            float learning_rate, float momentum, float) {
      if constexpr (module_t::input) {
        std::mt19937 gen(42); // NOLINT
        auto lr = learning_rate;

        for (size_t e = 0; e < EPOCHS; ++e) {
          std::cout << "Epoch " << e + 1 << "/" << EPOCHS << "\n";
          // lr *= (1.0f / (1.0f + decay * static_cast<float>(e)));

          std::shuffle(data_and_labels.begin(), data_and_labels.end(), gen);

          for (size_t batch_index = 0; batch_index < NUM_SAMPLES; batch_index += BATCH_SIZE)
            update_mini_batch(data_and_labels, batch_index, std::min(NUM_SAMPLES, batch_index + BATCH_SIZE), lr, momentum);
        }
      }
    }

    template <size_t NUM_SAMPLES, size_t EPOCHS, size_t BATCH_SIZE, typename predict_t>
    void fit(std::vector<std::tuple<input_t, predict_t>> &data_and_labels,
            float learning_rate, float momentum, float decay) {
      SGD<NUM_SAMPLES, EPOCHS, BATCH_SIZE>(data_and_labels, learning_rate,
                                          momentum, decay);
    }

    auto predict(const input_t &input) { return forward(input).imax(); }

    template <typename input_t, typename predict_t>
    auto &predict(const std::vector<input_t> &data, std::vector<predict_t> &out) {
      for (const input_t &input : data) out.push_back(predict(input));
      return out;
    }
  };

template <>
struct MLP<> {
  explicit MLP(std::mt19937 &) {};

  template <typename predict_t>
  inline const predict_t &forward(const predict_t &input) { return input; }

  template <typename predict_t>
  inline const predict_t &backward(const predict_t &, const predict_t &predict) { return predict; }
  
  inline void zero_grad() {}
  inline void update(float, float) {}
};

} // namespace nn

#endif  // NEUWURONKA_NETWORK_NEW_HPP
