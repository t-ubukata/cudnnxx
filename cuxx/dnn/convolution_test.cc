#include <vector>

#include "gtest/gtest.h"
#include "cuxx/dnn/convolution.h"

namespace cuxx {
namespace dnn {

class ConvolutionTest : public ::testing::Test {
 protected:
  Handle handle;
};

TEST_F(ConvolutionTest, TestConstructor2d) {
  Convolution<float, float> conv(4, 4, 2, 2, 2, 2, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
}

TEST_F(ConvolutionTest, TestConstructorNd) {
  constexpr int n = 3;
  constexpr int pads[n] = {4, 4, 4};
  constexpr int filter_strides[n] = {2, 2, 2};
  constexpr int dilations[n] = {2, 2, 2};
  Convolution<float, float> conv(n, pads, filter_strides, dilations,
                                 CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);
}

TEST_F(ConvolutionTest, TestGroupCount) {
  Convolution<float, float> conv(4, 4, 2, 2, 2, 2, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  constexpr int count = 2;
  conv.SetGroupCount(count);
  EXPECT_EQ(count, conv.GetGroupCount()) << "Group count mismatch.";
}

TEST_F(ConvolutionTest, TestMathType) {
  Convolution<float, float> conv(4, 4, 2, 2, 2, 2, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  constexpr cudnnMathType_t type = CUDNN_DEFAULT_MATH;
  conv.SetMathType(type);
  EXPECT_EQ(type, conv.GetMathType()) << "Math type mismatch.";
}

// No value check.
TEST_F(ConvolutionTest, TestGetForwardAlgorithmMaxCount) {
  Convolution<float, float> conv(4, 4, 2, 2, 2, 2, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  int count = conv.GetForwardAlgorithmMaxCount(handle);
  CUXX_UNUSED_VAR(count);
  SUCCEED();
}

// No value check.
TEST_F(ConvolutionTest, TestGetForwardAlgorithm) {
  Convolution<float, float> conv(4, 4, 2, 2, 2, 2, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  constexpr int x_n = 2;
  constexpr int x_c = 3;
  constexpr int x_h = 32;
  constexpr int x_w = 32;
  constexpr int n_x_elem = x_n * x_c * x_h * x_w;
  size_t x_size = sizeof(float) * n_x_elem;
  float* x_host[n_x_elem] = {};
  float* x_dev = nullptr;
  cudaMalloc(&x_dev, x_size);
  cudaMemcpy(x_host, x_dev, x_size, cudaMemcpyHostToDevice);
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, x_n, x_c, x_h, x_w, x_dev);

  constexpr int w_k = 2;  // the number of output feature maps.
  constexpr int w_c = 3;  // the number of input feature maps.
  constexpr int w_h = 5;  // The height of each filter.
  constexpr int w_w = 5;  // The width of each filter.
  constexpr int n_w_elem = w_k * w_c * w_h * w_w;
  size_t w_size = sizeof(float) * n_w_elem;
  float* w_host[n_w_elem] = {};
  float* w_dev = nullptr;
  cudaMalloc(&w_dev, w_size);
  cudaMemcpy(w_host, w_dev, w_size, cudaMemcpyHostToDevice);
  const Filter<float> w(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, w_k, w_c, w_h, w_w, w_dev);

  constexpr int y_n = 2;
  constexpr int y_c = 1;
  constexpr int y_h = 28;
  constexpr int y_w = 28;
  constexpr int n_y_elem = y_n * y_c * y_h * y_w;
  size_t y_size = sizeof(float) * n_y_elem;
  float* y_host[n_y_elem] = {};
  float* y_dev = nullptr;
  cudaMalloc(&y_dev, y_size);
  cudaMemcpy(y_host, y_dev, y_size, cudaMemcpyHostToDevice);
  Tensor<float> y(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, y_n, y_c, y_h, y_w, y_dev);

  int requested_count = conv.GetForwardAlgorithmMaxCount(handle);
  int returned_count = 0;
  std::vector<cudnnConvolutionFwdAlgoPerf_t> results_vec(requested_count);
  conv.GetForwardAlgorithm(handle, x, w, y, requested_count, &returned_count,
                           results_vec.data());
  cudaFree(x_dev);
  EXPECT_EQ(requested_count, returned_count) << "Count does not match.";
}

}  // namespace dnn
}  // namespace cuxx
