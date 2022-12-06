//
// Created by Martin Krebs on 15/10/2022.
//

#include <fstream>
#include <iostream>
#include <vector>

#include "nn/mlp.hpp"

template <typename input_t, typename output_t, size_t NUM_SAMPLES>
static std::vector<std::tuple<input_t, output_t>> &load_train_data_and_labels(
    const char *const &data_path, const char *const &labels_path,
    std::vector<std::tuple<input_t, output_t>> &out)
{
  std::ifstream data(data_path);
  std::ifstream labels(labels_path);

  size_t pixel;
  size_t label;
  char delim;

  for (size_t i = 0; i < NUM_SAMPLES; ++i)
  {
    out.emplace_back();
    for (size_t j = 0; j < input_t::size - 1; ++j)
    {
      data >> pixel;
      data >> delim;
      std::get<0>(out[i])[j] = static_cast<float>(pixel) / 255.0;
    }
    data >> pixel;
    data >> std::ws;
    std::get<0>(out[i])[input_t::size - 1] = static_cast<float>(pixel) / 255.0;

    labels >> label;
    labels >> std::ws;
    std::get<1>(out[i])[label] = 1.0;
  }

  data.close();
  labels.close();

  return out;
}

template <typename V, size_t N>
static std::vector<V> &load_test_data(const char *const data_path,
                                      std::vector<V> &out)
{
  std::ifstream data(data_path);

  size_t pixel;
  char delim;

  for (size_t i = 0; i < N; ++i)
  {
    out.emplace_back();
    for (size_t j = 0; j < V::size - 1; ++j)
    {
      data >> pixel;
      data >> delim;
      out[i].vector[j] = static_cast<float>(pixel) / 255.0;
    }
    data >> pixel;
    data >> std::ws;
    out[i].vector[V::size - 1] = static_cast<float>(pixel) / 255.0;
  }

  data.close();

  return out;
}

static void mnist_network()
{
  constexpr size_t INPUT_IMAGE_WIDTH = 28;
  constexpr size_t INPUT_IMAGE_HEIGHT = 28;
  constexpr size_t INPUT_FEATURES = INPUT_IMAGE_HEIGHT * INPUT_IMAGE_WIDTH;
  constexpr size_t OUTPUT_FEATURES = 10;

  constexpr size_t TRAIN_SAMPLE_SIZE = 60000;
  constexpr size_t TEST_SAMPLE_SIZE = 10000;

  constexpr auto TRAIN_DATA_PATH = "../data/fashion_mnist_train_vectors.csv";
  constexpr auto TRAIN_LABELS_PATH = "../data/fashion_mnist_train_labels.csv";

  constexpr auto TEST_DATA_PATH = "../data/fashion_mnist_test_vectors.csv";
  constexpr auto TEST_LABELS_PATH = "../data/fashion_mnist_test_labels.csv";

  std::mt19937 gen(42);

  std::vector<std::tuple<Vector<INPUT_FEATURES>, Vector<OUTPUT_FEATURES>>>
      train_data_and_labels;
  std::vector<Vector<INPUT_FEATURES>> test_data;
  std::vector<int> test_predictions;
  train_data_and_labels.reserve(TRAIN_SAMPLE_SIZE);
  test_data.reserve(TEST_SAMPLE_SIZE);
  test_predictions.reserve(TEST_SAMPLE_SIZE);

  std::cout << "Loading data and labels...\n";

  load_train_data_and_labels<Vector<INPUT_FEATURES>, Vector<OUTPUT_FEATURES>,
                             TRAIN_SAMPLE_SIZE>(
      TRAIN_DATA_PATH, TRAIN_LABELS_PATH, train_data_and_labels);

  load_test_data<Vector<INPUT_FEATURES>, TEST_SAMPLE_SIZE>(TEST_DATA_PATH,
                                                           test_data);


  auto mnist_network = nn::MLP<nn::Linear<INPUT_FEATURES, 90, true>, nn::Linear<90, OUTPUT_FEATURES>>(gen);

  std::cout << "Training network...\n";

  mnist_network.fit<TRAIN_SAMPLE_SIZE, 7, 64>(train_data_and_labels, 0.15f, 0.5f, 0.01f);

  std::cout << "Predicting...\n";

  std::ofstream train_predictions("../train_predictions.csv");
  int train_correct = 0;

  for (auto const &[data, label] : train_data_and_labels)
  {
    train_predictions << mnist_network.predict(data) << "\n";
    auto prediction = mnist_network.predict(data);
    if (label[prediction] == 1.0f)
      ++train_correct;
    train_predictions << prediction << "\n";
  }

  train_predictions.close();

  mnist_network.predict(test_data, test_predictions);

  std::ofstream test_predictions_file("../test_predictions.csv");

  for (int i : test_predictions)
    test_predictions_file << i << "\n";

  test_predictions_file.close();

  std::ifstream labels(TEST_LABELS_PATH);
  float correct = 0;
  int label;

  for (int i : test_predictions)
  {
    labels >> label;
    labels >> std::ws;
    if (label == i)
      ++correct;
  }

  labels.close();

  std::cout << "Train accuracy: " << train_correct / static_cast<float>(TRAIN_SAMPLE_SIZE)
  << "\n";

  std::cout << "Test acccuracy: " << correct / static_cast<float>(TEST_SAMPLE_SIZE) << "\n";

  std::cout << "Done!\n";
}

int main()
{
  mnist_network();

  return 0;
}
