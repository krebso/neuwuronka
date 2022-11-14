//
// Created by Martin Krebs on 15/10/2022.
//

#include <iostream>

#include "matrix.hpp"
#include "vector.hpp"
#include "network.hpp"
#include "math.hpp"
#include "layer.hpp"


int main() {

    /*
     * constexpr size_t INPUT_IMAGE_WIDTH = 28;
     * constexpr size_t INPUT_IMAGE_HEIGHT = 28;
     * constexpr size_t INPUT_DIMENSION = INPUT_IMAGE_HEIGHT * INPUT_IMAGE_WIDTH;

     * constexpr size_t TRAIN_SAMPLE_SIZE = 60000;
     * constexpr size_t TEST_SAMPLE_SIZE = 10000;
     */

    std::cout << "Hello, Neuwuronka\n";

    /*
    std::array<std::tuple<float, float>, 4> train_X = {  };
    train_X[0] = {1, 0};
    train_X[1] = {0, 1};
    train_X[2] = {0, 0};
    train_X[3] = {1, 1};

    std::array<int, 4> train_y = {1, 1, 0, 0};
    */

    // create network
    auto xor_network = Network<InputLayer<2>, OutputLayer<2>>();

    // auto x = xor_network.predict(Vector<2>{1, 1});
    auto x = xor_network.predict(Vector<2>{0, 0});

    // preprocess the data

    // train the network
    // xor_network.fit()

    // predict the result
    // xor_network.predict()

    // save the result

    return 0;
}
