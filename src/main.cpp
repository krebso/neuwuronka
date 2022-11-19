//
// Created by Martin Krebs on 15/10/2022.
//

#include <fstream>
#include <iostream>
#include <vector>

#include "layer.hpp"
#include "math.hpp"
#include "matrix.hpp"
#include "network.hpp"
#include "vector.hpp"

/* All the different sources I used when writing this project
 * [1] http://neuralnetworksanddeeplearning.com/chap1.html
 * [2] https://gist.github.com/alexshtf/eb5128b3e3e143187794
 * [3] https://github.com/mnielsen/neural-networks-and-deep-learning
 */

/*
 * Agenda
 * [1] fix algorithm
 */

template <typename input_t, typename output_t, size_t NUM_SAMPLES>
std::vector<std::tuple<input_t, output_t>>& load_train_data_and_labels(
    const char* const& data_path, const char* const& labels_path, std::vector<std::tuple<input_t, output_t>>& out) {
    std::ifstream data(data_path);
    std::ifstream labels(labels_path);

    size_t pixel;
    size_t label;
    char delim;

    for (size_t i = 0; i < NUM_SAMPLES; ++i) {
        out.emplace_back();
        for (size_t j = 0; j < input_t::size - 1; ++j) {
            data >> pixel;
            data >> delim;
            std::get<0>(out[i])[j] = static_cast<float>(pixel) / 256.0;
        }
        data >> pixel;
        data >> std::ws;
        std::get<0>(out[i])[input_t::size - 1] = static_cast<float>(pixel) / 256.0;

        labels >> label;
        labels >> std::ws;
        std::get<1>(out[i])[label] = 1.0;
    }

    data.close();
    labels.close();

    return out;
}

template <typename V, size_t N>
std::vector<V>& load_test_data(const char* const data_path, std::vector<V>& out) {
    std::ifstream data(data_path);

    size_t pixel;
    char delim;

    for (size_t i = 0; i < N; ++i) {
        out.emplace_back();
        for (size_t j = 0; j < V::size - 1; ++j) {
            data >> pixel;
            data >> delim;
            out[i].vector[j] = static_cast<float>(pixel) / 256.0;
        }
        data >> pixel;
        data >> std::ws;
        out[i].vector[V::size - 1] = static_cast<float>(pixel) / 256.0;
    }

    data.close();

    return out;
}

void xor_network() {
    std::vector<std::tuple<Vector<2>, Vector<2>>> xor_data_and_labels;

    xor_data_and_labels.push_back({{1, 1}, {1, 0}});
    xor_data_and_labels.push_back({{0, 0}, {1, 0}});
    xor_data_and_labels.push_back({{1, 0}, {0, 1}});
    xor_data_and_labels.push_back({{0, 1}, {0, 1}});

    std::mt19937 gen(42);  // NOLINT

    auto xor_network = Network<InputLayer<2>, HiddenLayer<5>, OutputLayer<2>>(gen);

    xor_network.fit<4, 1000, 3>(xor_data_and_labels);

    auto x = xor_network.predict(Vector<2>{1, 1});
    std::cout << "Predicted: [1, 1] -> " << x << "\n";
    x = xor_network.predict(Vector<2>{0, 0});
    std::cout << "Predicted: [0, 0] -> " << x << "\n";
    x = xor_network.predict(Vector<2>{0, 1});
    std::cout << "Predicted: [0, 1] -> " << x << "\n";
    x = xor_network.predict(Vector<2>{1, 0});
    std::cout << "Predicted: [1, 0] -> " << x << "\n";
}

void mnist_network() {
    constexpr size_t INPUT_IMAGE_WIDTH = 28;
    constexpr size_t INPUT_IMAGE_HEIGHT = 28;
    constexpr size_t INPUT_DIMENSION = INPUT_IMAGE_HEIGHT * INPUT_IMAGE_WIDTH;
    constexpr size_t OUTPUT_DIMENSION = 10;

    constexpr size_t TRAIN_SAMPLE_SIZE = 60000;
    constexpr size_t TEST_SAMPLE_SIZE = 10000;

    constexpr auto TRAIN_DATA_PATH = "../data/fashion_mnist_train_vectors.csv";
    constexpr auto TRAIN_LABELS_PATH = "../data/fashion_mnist_train_labels.csv";

    constexpr auto TEST_DATA_PATH = "../data/fashion_mnist_test_vectors.csv";
    constexpr auto TEST_LABELS_PATH = "../data/fashion_mnist_test_labels.csv";

    std::mt19937 gen(42);

    std::vector<std::tuple<Vector<INPUT_DIMENSION>, Vector<OUTPUT_DIMENSION>>> train_data_and_labels;
    train_data_and_labels.reserve(TRAIN_SAMPLE_SIZE);
    load_train_data_and_labels<Vector<INPUT_DIMENSION>, Vector<OUTPUT_DIMENSION>, TRAIN_SAMPLE_SIZE>(
        TRAIN_DATA_PATH, TRAIN_LABELS_PATH, train_data_and_labels);

    std::vector<Vector<INPUT_DIMENSION>> test_data;
    std::vector<int> test_predictions;

    test_data.reserve(TEST_SAMPLE_SIZE);
    test_predictions.reserve(TEST_SAMPLE_SIZE);

    load_test_data<Vector<INPUT_DIMENSION>, TEST_SAMPLE_SIZE>(TEST_DATA_PATH, test_data);

    auto mnist_network = Network<InputLayer<INPUT_DIMENSION>, HiddenLayer<64>, OutputLayer<OUTPUT_DIMENSION>>(gen);

    mnist_network.fit<TRAIN_SAMPLE_SIZE, 10, 128>(train_data_and_labels);

    mnist_network.predict(test_data, test_predictions);

    std::ofstream predictions_file("../data/predictions.csv");

    for (int i : test_predictions) predictions_file << i << "\n";

    predictions_file.close();
}

int main() {
    // xor_network();
    mnist_network();

    return 0;
}
