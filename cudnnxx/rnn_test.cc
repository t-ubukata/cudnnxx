#include "cudnnxx/dropout.h"
#include "cudnnxx/rnn.h"
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
  RNN<float, float> rnn(handle, 1, 1, dropout, CUDNN_LINEAR_INPUT, CUDNN_UNIDIRECTIONAL,
  CUDNN_RNN_RELU, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT);
}

TEST_F(RNNTest, TestGetParamSize) {
  float dropout_p = 0.5;
  unsigned long long seed = 20200627;
  Dropout<float> dropout(handle, dropout_p, seed);
  auto dtype = CUDNN_DATA_FLOAT;
  RNN<float, float> rnn(handle, 1, 1, dropout, CUDNN_LINEAR_INPUT, CUDNN_UNIDIRECTIONAL,
  CUDNN_RNN_RELU, CUDNN_RNN_ALGO_STANDARD, dtype);

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
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(x_dev, x_host, size, cudaMemcpyHostToDevice));
  Tensor<float> x_tensor(dtype, CUDNN_TENSOR_NCHW, n, c, h, w, x_dev);

  auto size_in_bytes = rnn.GetParamsSize(handle, x_tensor, dtype);

  size_t size_in_bytes_ref = 0;
  CUDNNXX_DNN_CHECK(cudnnGetRNNParamsSize(handle.raw_handle(), rnn.desc(), x_tensor.desc(), &size_in_bytes_ref, dtype));

  EXPECT_EQ(size_in_bytes_ref, size_in_bytes);
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
}

}  // namespace cudnnxx
