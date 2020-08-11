#include "cudnnxx/rnn.h"

#include <vector>

#include "cudnnxx/dropout.h"
#include "gtest/gtest.h"

namespace cudnnxx {

class RNNTest : public ::testing::Test {
 protected:
  Handle handle;
};

TEST_F(RNNTest, TestConstructor) {
  float dropout_p = 0.5;
  unsigned long long seed = 20200627;
  Dropout<float> dropout(handle, dropout_p, seed);
  RNN<float, float> rnn(handle, 1, 1, dropout, CUDNN_LINEAR_INPUT,
                        CUDNN_UNIDIRECTIONAL, CUDNN_RNN_RELU,
                        CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT);
}

TEST_F(RNNTest, TestGetParamSize) {
  float dropout_p = 0.5;
  unsigned long long seed = 20200627;
  Dropout<float> dropout(handle, dropout_p, seed);
  auto dtype = CUDNN_DATA_FLOAT;
  RNN<float, float> rnn(handle, 1, 1, dropout, CUDNN_LINEAR_INPUT,
                        CUDNN_UNIDIRECTIONAL, CUDNN_RNN_RELU,
                        CUDNN_RNN_ALGO_STANDARD, dtype);

  constexpr int n = 2;
  constexpr int c = 3;
  constexpr int h = 2;
  constexpr int w = 2;
  constexpr int n_elem = n * c * h * w;
  float x_host[n_elem] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                          0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                          1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4};
  float* x_dev = nullptr;
  size_t size = sizeof(float) * n_elem;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, size));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host, size, cudaMemcpyHostToDevice));
  Tensor<float> x_tensor(dtype, CUDNN_TENSOR_NCHW, n, c, h, w, x_dev);

  auto size_in_bytes = rnn.GetParamsSize(handle, x_tensor, dtype);

  size_t size_in_bytes_ref = 0;
  CUDNNXX_DNN_CHECK(cudnnGetRNNParamsSize(handle.raw_handle(), rnn.desc(),
                                          x_tensor.desc(), &size_in_bytes_ref,
                                          dtype));

  EXPECT_EQ(size_in_bytes_ref, size_in_bytes);
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
}

TEST_F(RNNTest, TestGetRNNTrainingReserveSize) {
  float dropout_p = 0.5;
  unsigned long long seed = 20200627;
  Dropout<float> dropout(handle, dropout_p, seed);
  auto dtype = CUDNN_DATA_FLOAT;
  RNN<float, float> rnn(handle, 1, 1, dropout, CUDNN_LINEAR_INPUT,
                        CUDNN_UNIDIRECTIONAL, CUDNN_RNN_RELU,
                        CUDNN_RNN_ALGO_STANDARD, dtype);

  constexpr int n_dims = 3;
  constexpr int batch_size = 2;
  constexpr int input_size = 3;
  int dims[n_dims] = {batch_size, input_size, 1};
  int strides[n_dims] = {input_size, 1, 1};
  constexpr int n_elem = batch_size * input_size * 1;
  constexpr int seq_length = 3;
  float xs_host[seq_length][n_elem] = {{0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
                                       {0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
                                       {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}};
  float* xs_dev[seq_length] = {nullptr};
  size_t x_size = sizeof(float) * n_elem;

  std::vector<Tensor<float>> x_tensors;
  x_tensors.reserve(seq_length);
  for (int i = 0; i < seq_length; ++i) {
    CUDNNXX_CUDA_CHECK(cudaMalloc(&xs_dev[i], x_size));
    CUDNNXX_CUDA_CHECK(
        cudaMemcpy(xs_dev[i], xs_host[i], x_size, cudaMemcpyHostToDevice));
    x_tensors.emplace_back(dtype, n_dims, dims, strides, xs_dev[i]);
  }

  auto reserve_size = rnn.GetTrainingReserveSize(handle, seq_length, x_tensors);

  std::vector<cudnnTensorDescriptor_t> x_descs;
  for (int i = 0; i < seq_length; ++i) {
    x_descs.push_back(x_tensors[i].desc());
  }
  size_t reserve_size_ref = 0;
  CUDNNXX_DNN_CHECK(cudnnGetRNNTrainingReserveSize(
      handle.raw_handle(), rnn.desc(), seq_length, x_descs.data(),
      &reserve_size_ref));

  EXPECT_EQ(reserve_size_ref, reserve_size);
  for (int i = 0; i < seq_length; ++i) {
    CUDNNXX_CUDA_CHECK(cudaFree(xs_dev[i]));
  }
}

}  // namespace cudnnxx
