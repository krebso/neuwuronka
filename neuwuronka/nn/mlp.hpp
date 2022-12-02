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

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif

namespace nn {

// convenience wrapper, so I can use the torch style
struct Module {};
struct AutoGradFunction : Module {};

struct ReLU : AutoGradFunction {
  static constexpr auto f = relu<float>;
  static constexpr auto f_prime = relu_prime<float>;
};

// Each MLP is constructed using layers
template <size_t NEURONS, bool _input = false, bool _output = false>
struct Layer : Module {
  static constexpr size_t size = NEURONS;
  static constexpr bool input = _input;
  static constexpr bool output = _output;
};

// Interface for the whole network
template <typename... args>
class MLP;
template <typename previous_layer, typename activation_function,
          typename current_layer, typename... args>
struct MLP<previous_layer, activation_function, current_layer, args...> {
  using network_t = MLP<current_layer, args...>;
  using input_t   = Vector<previous_layer::size>;
  using output_t  = Vector<current_layer::size>;
  using predict_t = typename network_t::predict_t;
  using weights_t = Matrix<current_layer::size, previous_layer::size>;
  using biases_t  = output_t;

  network_t network;  // rest of the recursively defined network

  output_t  weighted_input;  // allows us to store inner potential vectors which
                            // will be used during backward pass
  output_t  activation;      // vector which is being passed to the next layer as
                            // an input, also used during backward pass

  output_t  activation_prime;  // output vector for computing first derivative of
                              // activation function

  weights_t weights;  // matrix of weights
  biases_t  biases;    // vector of biases

  weights_t grad_w;  // this is our aggregator for storing the deltas through
                      // one iteration of SGD
  biases_t  grad_b;   // same as above, but for biases

  weights_t momentum_w;  // momentum weights history
  biases_t  momentum_b;   // same as above for biases

  weights_t delta_grad_w;  // pre-allocated output parameter, used during
                            // back-prop for storing single delta
  biases_t  delta_grad_b;   // same as above

  template <typename D, typename G>
  void init_weights_and_biases(D &distribution, G &generator) {
    // given desired initial weight distribution, init weights and biases
    for (size_t h = 0; h < weights.height; ++h) {
      for (size_t w = 0; w < weights.width; ++w)
        weights.at(h, w) = distribution(generator);
      biases[h] = distribution(generator);
    }
  }

  void zero_grad() {
    grad_b.zero();
    grad_w.zero();
    if constexpr (!current_layer::output) network.zero_grad();
  }

  void update(float learning_rate, float momentum) {
    // update the friction
    momentum_w *momentum;
    momentum_b *momentum;

    // update history with current nablas
    momentum_w += grad_w * learning_rate;
    momentum_b += grad_b * learning_rate;

    // update weights and biases
    weights -= momentum_w;
    biases -= momentum_b;

    // update the rest of the network
    if constexpr (!current_layer::output)
      network.update(learning_rate, momentum);
  }

  void update_mini_batch(
      std::vector<std::tuple<input_t, predict_t>> &data_and_labels,
      size_t start, size_t end, float learning_rate, float momentum) {
    // num of samples in minibatch
    auto n = static_cast<float>(end - start);

    // recursively zero out both gradients
    zero_grad();

    // for each training sample
    for (size_t i = start; i < end; ++i) {
      auto x = std::get<0>(data_and_labels[i]);

      // set activations, (both actual and prime) and weighted_inputs
      forward(x);

      // compute the gradient using backward pass
      backward(std::get<0>(data_and_labels[i]),
               std::get<1>(data_and_labels[i]));
    }

    // update the weights and biases recursively for whole network
    update(learning_rate / n, momentum);
  }

  template <size_t NUM_SAMPLES, size_t EPOCHS, size_t BATCH_SIZE>
  void SGD(std::vector<std::tuple<input_t, predict_t>> &data_and_labels,
           float learning_rate, float momentum, float decay) {
    if (previous_layer::input) {
      std::mt19937 gen(42);  // NOLINT
      auto lr = learning_rate;

      // in each epoch
      for (size_t e = 0; e < EPOCHS; ++e) {
        std::cout << "Epoch " << e + 1 << "/" << EPOCHS << "\n";
        lr *= (1.0f / (1.0f + decay * static_cast<float>(e)));

        // shuffle the data randomly
        std::shuffle(data_and_labels.begin(), data_and_labels.end(), gen);

        // for all mini-batches
        for (size_t batch_index = 0; batch_index < NUM_SAMPLES;
             batch_index += BATCH_SIZE)

          // perform gradient descent on single minibatch
          update_mini_batch(data_and_labels, batch_index,
                            std::min(NUM_SAMPLES, batch_index + BATCH_SIZE), lr,
                            momentum);
      }
    }
  }

  const biases_t &backward(const input_t &input, const predict_t &predict) {
    if constexpr (current_layer::output) {
      cross_entropy_cost_function_prime(activation, predict, delta_grad_b);
    } else {
      // get the delta from next layer
      auto &next_delta_nabla_b = network.backward(activation, predict);

      // multiply the corresponding matrix
      dot_matrix_transposed_vector(network.weights, next_delta_nabla_b,
                                   delta_grad_b);
    }

    // multiply with the derivative of activation function
    delta_grad_b *= activation_prime;

    // compute the delta for weights
    dot_vector_transposed_vector(delta_grad_b, input, delta_grad_w);

    // update the aggregator for minibatch
    grad_b += delta_grad_b;
    grad_w += delta_grad_w;

    // send the delta for layer below
    return delta_grad_b;
  }

  predict_t forward(const input_t &input) {
    // compute weighted input
    dot_matrix_vector_transposed(weights, input, weighted_input);
    weighted_input += biases;

    // apply activation function
    map(activation_function::f, weighted_input, activation);

    // pre-compute first derivative of activation function, will be used during
    // backward pass
    map(activation_function::f_prime, weighted_input, activation_prime);

    // pass the activation as an input to the next layer
    return network.forward(activation);
  }

  explicit MLP(std::mt19937 &gen)
      : network(gen),
        weighted_input(),
        activation(),
        activation_prime(),
        weights(),
        biases(),
        grad_w(),
        grad_b(),
        delta_grad_w(),
        delta_grad_b(),
        momentum_w(),
        momentum_b() {
    if constexpr (false || current_layer::output) {  // Xavier
      float lower = -(1.0 / std::sqrt(previous_layer::size));
      float upper = (1.0 / std::sqrt(previous_layer::size));
      auto distribution = std::uniform_real_distribution<float>(lower, upper);
      init_weights_and_biases(distribution, gen);
    } else {  // He
      auto distribution = std::normal_distribution<float>(
          0.0, std::sqrt(2.0 / previous_layer::size));
      init_weights_and_biases(distribution, gen);
    }
  }

  template <size_t NUM_SAMPLES, size_t EPOCHS, size_t BATCH_SIZE>
  void fit(std::vector<std::tuple<input_t, predict_t>> &data_and_labels,
           float learning_rate, float momentum, float decay) {
    SGD<NUM_SAMPLES, EPOCHS, BATCH_SIZE>(data_and_labels, learning_rate,
                                         momentum, decay);
  }

  auto predict(const input_t &v) { return forward(v).imax(); }

  template <typename input_t, typename predict_t>
  auto &predict(const std::vector<input_t> &data, std::vector<predict_t> &out) {
    for (const input_t &input : data) out.push_back(predict(input));
    return out;
  }
};

template <typename output_layer>
struct MLP<output_layer> {
  using predict_t = Vector<output_layer::size>;

  predict_t activation;

  explicit MLP(std::mt19937 &) : activation(){};

  predict_t forward(const predict_t &input) {
    return softmax(input, activation);
  }
};
}  // namespace nn

#endif  // NEUWURONKA_NETWORK_HPP

template <size_t S>
inline auto &cross_entropy_cost_function_prime(const Vector<S> &activation,
                                               const Vector<S> &y,
                                               Vector<S> &out) {
  for (size_t i = 0; i < S; ++i)
    out.vector[i] = activation.vector[i] - y.vector[i];
  return out;
}