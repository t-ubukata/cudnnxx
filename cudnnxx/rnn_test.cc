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

  constexpr int n_dims = 3;
  constexpr int batch_size = 2;
  constexpr int input_size = 3;
  int dims[n_dims] = {batch_size, input_size, 1};
  int strides[n_dims] = {dims[2] * dims[1], dims[2], 1};
  constexpr int n_elem = input_size * batch_size;
  float x_host[n_elem] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};

  float* x_dev = nullptr;
  size_t size = sizeof(float) * n_elem;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, size));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host, size, cudaMemcpyHostToDevice));
  Tensor<float> x_tensor(dtype, n_dims, dims, strides, x_dev);

  auto size_in_bytes = rnn.GetParamsSize(handle, x_tensor, dtype);

  size_t size_in_bytes_ref = 0;
  CUDNNXX_DNN_CHECK(cudnnGetRNNParamsSize(handle.raw_handle(), rnn.desc(),
                                          x_tensor.desc(), &size_in_bytes_ref,
                                          dtype));

  EXPECT_EQ(size_in_bytes_ref, size_in_bytes);
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
}

TEST_F(RNNTest, TestGetTrainingReserveSize) {
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
  int strides[n_dims] = {dims[2] * dims[1], dims[2], 1};
  constexpr int seq_length = 3;
  constexpr int n_elem = seq_length * input_size * batch_size;

  float x_host[n_elem] = {};
  float* x_dev = nullptr;
  size_t x_size = sizeof(float) * n_elem;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, x_size));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host, x_size, cudaMemcpyHostToDevice));
  TensorArray<float> x_tensors(dtype, n_dims, dims, strides, x_dev, seq_length);

  auto reserve_size = rnn.GetTrainingReserveSize(handle, seq_length, x_tensors);

  size_t reserve_size_ref = 0;
  CUDNNXX_DNN_CHECK(cudnnGetRNNTrainingReserveSize(
      handle.raw_handle(), rnn.desc(), seq_length, x_tensors.descs(),
      &reserve_size_ref));

  EXPECT_EQ(reserve_size_ref, reserve_size);
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
}

TEST_F(RNNTest, TestGetRNNWorkspaceSize) {
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
  int strides[n_dims] = {dims[2] * dims[1], dims[2], 1};
  constexpr int seq_length = 3;
  constexpr int n_elem = seq_length * input_size * batch_size;

  float x_host[n_elem] = {};
  float* x_dev = nullptr;
  size_t x_size = sizeof(float) * n_elem;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, x_size));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host, x_size, cudaMemcpyHostToDevice));
  TensorArray<float> x_tensors(dtype, n_dims, dims, strides, x_dev, seq_length);

  auto reserve_size = rnn.GetWorkspaceSize(handle, seq_length, x_tensors);

  size_t reserve_size_ref = 0;
  CUDNNXX_DNN_CHECK(cudnnGetRNNWorkspaceSize(handle.raw_handle(), rnn.desc(),
                                             seq_length, x_tensors.descs(),
                                             &reserve_size_ref));

  EXPECT_EQ(reserve_size_ref, reserve_size);
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
}

}  // namespace cudnnxx
