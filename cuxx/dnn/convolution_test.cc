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
  EXPECT_EQ(count, conv.GetGroupCount()) << "Group does not match.";
}

TEST_F(ConvolutionTest, TestMathType) {
  Convolution<float, float> conv(4, 4, 2, 2, 2, 2, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  constexpr cudnnMathType_t type = CUDNN_DEFAULT_MATH;
  conv.SetMathType(type);
  EXPECT_EQ(type, conv.GetMathType()) << "Math type does not match.";
}

// No value check.
TEST_F(ConvolutionTest, TestGetForwardAlgorithmMaxCount) {
  Convolution<float, float> conv(4, 4, 2, 2, 2, 2, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  int count = conv.GetForwardAlgorithmMaxCount(handle);
  CUXX_UNUSED_VAR(count);
}

// No value check.
TEST_F(ConvolutionTest, TestGetForwardAlgorithm) {
  Convolution<float, float> conv(0, 0, 1, 2, 1, 1, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  constexpr int x_n = 32;
  constexpr int x_c = 3;
  constexpr int x_h = 6;
  constexpr int x_w = 4;
  constexpr int n_x_elem = x_n * x_c * x_h * x_w;
  size_t x_size = sizeof(float) * n_x_elem;
  float x_host[n_x_elem] = {};
  for (int i = 0; i < n_x_elem; ++i) {
    x_host[i] = i * 0.0001;
  }
  float* x_dev = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&x_dev, x_size));
  CUXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host, x_size, cudaMemcpyHostToDevice));
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, x_n, x_c, x_h, x_w,
                  x_dev);

  constexpr int w_k = 4;  // The number of output feature maps.
  constexpr int w_c = 3;  // The number of input feature maps.
  constexpr int w_h = 3;  // The height of each filter.
  constexpr int w_w = 3;  // The width of each filter.
  constexpr int n_w_elem = w_k * w_c * w_h * w_w;
  size_t w_size = sizeof(float) * n_w_elem;
  float w_host[n_w_elem] = {};
  for (int i = 0; i < n_w_elem; ++i) {
    w_host[i] = i * 0.00001;
  }
  float* w_dev = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&w_dev, w_size));
  CUXX_CUDA_CHECK(cudaMemcpy(w_dev, w_host, w_size, cudaMemcpyHostToDevice));
  Filter<float> w(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, w_k, w_c, w_h, w_w,
                  w_dev);

  constexpr int y_n = 32;
  constexpr int y_c = 4;
  constexpr int y_h = 4;
  constexpr int y_w = 1;
  constexpr int n_y_elem = y_n * y_c * y_h * y_w;
  size_t y_size = sizeof(float) * n_y_elem;
  float* y_dev = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&y_dev, y_size));
  Tensor<float> y(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, y_n, y_c, y_h, y_w,
                  y_dev);

  int requested_count = conv.GetForwardAlgorithmMaxCount(handle);
  int returned_count = 0;
  std::vector<cudnnConvolutionFwdAlgoPerf_t> results_vec(requested_count);
  conv.GetForwardAlgorithm(handle, x, w, y, requested_count, &returned_count,
                           results_vec.data());

  CUXX_CUDA_CHECK(cudaFree(y_dev));
  CUXX_CUDA_CHECK(cudaFree(w_dev));
  CUXX_CUDA_CHECK(cudaFree(x_dev));
  EXPECT_EQ(requested_count, returned_count) << "Count does not match.";
}

// No value check.
TEST_F(ConvolutionTest, TestGetForwardWorkspaceSize) {
  Convolution<float, float> conv(0, 0, 1, 2, 1, 1, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  constexpr int x_n = 32;
  constexpr int x_c = 3;
  constexpr int x_h = 6;
  constexpr int x_w = 4;
  constexpr int n_x_elem = x_n * x_c * x_h * x_w;
  size_t x_size = sizeof(float) * n_x_elem;
  float x_host[n_x_elem] = {};
  float* x_dev = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&x_dev, x_size));
  CUXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host, x_size, cudaMemcpyHostToDevice));
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, x_n, x_c, x_h, x_w,
                  x_dev);

  constexpr int w_k = 4;  // The number of output feature maps.
  constexpr int w_c = 3;  // The number of input feature maps.
  constexpr int w_h = 3;  // The height of each filter.
  constexpr int w_w = 3;  // The width of each filter.
  constexpr int n_w_elem = w_k * w_c * w_h * w_w;
  size_t w_size = sizeof(float) * n_w_elem;
  float w_host[n_w_elem] = {};
  float* w_dev = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&w_dev, w_size));
  CUXX_CUDA_CHECK(cudaMemcpy(w_dev, w_host, w_size, cudaMemcpyHostToDevice));
  Filter<float> w(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, w_k, w_c, w_h, w_w,
                  w_dev);

  constexpr int y_n = 32;
  constexpr int y_c = 4;
  constexpr int y_h = 4;
  constexpr int y_w = 1;
  constexpr int n_y_elem = y_n * y_c * y_h * y_w;
  size_t y_size = sizeof(float) * n_y_elem;
  float* y_dev = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&y_dev, y_size));
  Tensor<float> y(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, y_n, y_c, y_h, y_w,
                  y_dev);

  const size_t size = conv.GetForwardWorkspaceSize(handle, x, w, y,
                               CUDNN_CONVOLUTION_FWD_ALGO_GEMM);
  CUXX_UNUSED_VAR(size);
  CUXX_CUDA_CHECK(cudaFree(y_dev));
  CUXX_CUDA_CHECK(cudaFree(w_dev));
  CUXX_CUDA_CHECK(cudaFree(x_dev));
}

// No value check.
TEST_F(ConvolutionTest, TestFindForwardAlgorithm) {
  Convolution<float, float> conv(0, 0, 1, 2, 1, 1, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  constexpr int x_n = 32;
  constexpr int x_c = 3;
  constexpr int x_h = 6;
  constexpr int x_w = 4;
  constexpr int n_x_elem = x_n * x_c * x_h * x_w;
  size_t x_size = sizeof(float) * n_x_elem;
  float x_host[n_x_elem] = {};
  float* x_dev = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&x_dev, x_size));
  CUXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host, x_size, cudaMemcpyHostToDevice));
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, x_n, x_c, x_h, x_w,
                  x_dev);

  constexpr int w_k = 4;  // The number of output feature maps.
  constexpr int w_c = 3;  // The number of input feature maps.
  constexpr int w_h = 3;  // The height of each filter.
  constexpr int w_w = 3;  // The width of each filter.
  constexpr int n_w_elem = w_k * w_c * w_h * w_w;
  size_t w_size = sizeof(float) * n_w_elem;
  float w_host[n_w_elem] = {};
  float* w_dev = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&w_dev, w_size));
  CUXX_CUDA_CHECK(cudaMemcpy(w_dev, w_host, w_size, cudaMemcpyHostToDevice));
  Filter<float> w(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, w_k, w_c, w_h, w_w,
                  w_dev);

  constexpr int y_n = 32;
  constexpr int y_c = 4;
  constexpr int y_h = 4;
  constexpr int y_w = 1;
  constexpr int n_y_elem = y_n * y_c * y_h * y_w;
  size_t y_size = sizeof(float) * n_y_elem;
  float* y_dev = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&y_dev, y_size));
  Tensor<float> y(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, y_n, y_c, y_h, y_w,
                  y_dev);

  const size_t ws_size = conv.GetForwardWorkspaceSize(handle, x, w, y,
                              CUDNN_CONVOLUTION_FWD_ALGO_GEMM);
  constexpr int requested_algo_count = 8;
  int returned_algo_count = 0;
  cudnnConvolutionFwdAlgoPerf_t results[requested_algo_count];
  void* ws = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&ws, ws_size));
  conv.FindForwardAlgorithm(handle, x, w, y, requested_algo_count,
                            &returned_algo_count, results, ws, ws_size);

  CUXX_CUDA_CHECK(cudaFree(ws));
  CUXX_CUDA_CHECK(cudaFree(y_dev));
  CUXX_CUDA_CHECK(cudaFree(w_dev));
  CUXX_CUDA_CHECK(cudaFree(x_dev));
}

TEST_F(ConvolutionTest, TestGet2dForwardOutputDim) {
  Convolution<float, float> conv(0, 0, 1, 2, 1, 1, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  constexpr int x_n = 32;
  constexpr int x_c = 3;
  constexpr int x_h = 6;
  constexpr int x_w = 4;
  constexpr int n_x_elem = x_n * x_c * x_h * x_w;
  size_t x_size = sizeof(float) * n_x_elem;
  float x_host[n_x_elem] = {};
  float* x_dev = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&x_dev, x_size));
  CUXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host, x_size, cudaMemcpyHostToDevice));
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, x_n, x_c, x_h, x_w,
                  x_dev);

  constexpr int f_k = 4;  // The number of output feature maps.
  constexpr int f_c = 3;  // The number of input feature maps.
  constexpr int f_h = 3;  // The height of each filter.
  constexpr int f_w = 3;  // The width of each filter.
  constexpr int n_f_elem = f_k * f_c * f_h * f_w;
  size_t f_size = sizeof(float) * n_f_elem;
  float f_host[n_f_elem] = {};
  float* f_dev = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&f_dev, f_size));
  CUXX_CUDA_CHECK(cudaMemcpy(f_dev, f_host, f_size, cudaMemcpyHostToDevice));
  Filter<float> f(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, f_k, f_c, f_h, f_w,
                  f_dev);

  int n = 0;
  int c = 0;
  int h = 0;
  int w = 0;
  conv.Get2dForwardOutputDim(x, f, &n, &c, &h, &w);
  EXPECT_EQ(32, n) << "n does not match.";
  EXPECT_EQ(4, c) << "c does not match.";
  EXPECT_EQ(4, h) << "h does not match.";
  EXPECT_EQ(1, w) << "w does not match.";

  CUXX_CUDA_CHECK(cudaFree(f_dev));
  CUXX_CUDA_CHECK(cudaFree(x_dev));
}

// TODO(t-ubukata): Test Nd.

TEST_F(ConvolutionTest, TestForward) {
  Convolution<float, float> conv(0, 0, 1, 2, 1, 1, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  constexpr int x_n = 32;
  constexpr int x_c = 3;
  constexpr int x_h = 6;
  constexpr int x_w = 4;
  constexpr int n_x_elem = x_n * x_c * x_h * x_w;
  size_t x_size = sizeof(float) * n_x_elem;
  float x_host[n_x_elem] = {};
  for (int i = 0; i < n_x_elem; ++i) {
    x_host[i] = i * 0.0001;
  }
  float* x_dev = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&x_dev, x_size));
  CUXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host, x_size, cudaMemcpyHostToDevice));
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, x_n, x_c, x_h, x_w,
                  x_dev);

  constexpr int w_k = 4;  // The number of output feature maps.
  constexpr int w_c = 3;  // The number of input feature maps.
  constexpr int w_h = 3;  // The height of each filter.
  constexpr int w_w = 3;  // The width of each filter.
  constexpr int n_w_elem = w_k * w_c * w_h * w_w;
  size_t w_size = sizeof(float) * n_w_elem;
  float w_host[n_w_elem] = {};
  for (int i = 0; i < n_w_elem; ++i) {
    w_host[i] = i * 0.00001;
  }
  float* w_dev = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&w_dev, w_size));
  CUXX_CUDA_CHECK(cudaMemcpy(w_dev, w_host, w_size, cudaMemcpyHostToDevice));
  Filter<float> w(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, w_k, w_c, w_h, w_w,
                  w_dev);

  constexpr int y_n = 32;
  constexpr int y_c = 4;
  constexpr int y_h = 4;
  constexpr int y_w = 1;
  constexpr int n_y_elem = y_n * y_c * y_h * y_w;
  size_t y_size = sizeof(float) * n_y_elem;
  float* y_dev = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&y_dev, y_size));
  Tensor<float> y(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, y_n, y_c, y_h, y_w,
                  y_dev);

  constexpr auto algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
  const size_t ws_size = conv.GetForwardWorkspaceSize(handle, x, w, y, algo);
  void* ws = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&ws, ws_size));

  constexpr float alpha = 1;
  constexpr float beta = 0;
  conv.Forward(handle, alpha, x, w, algo, ws, ws_size, beta, &y);
  float y_host[n_y_elem] = {};
  CUXX_CUDA_CHECK(cudaMemcpy(y_host, y_dev, y_size, cudaMemcpyDeviceToHost));

  float* y_dev_ref = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&y_dev_ref, y_size));
  Tensor<float> y_ref(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, y_n, y_c, y_h, y_w,
                      y_dev_ref);
  CUXX_DNN_CHECK(cudnnConvolutionForward(handle.raw_handle(),
                                         &alpha, x.desc(), x.dev_mem(),
                                         w.desc(), w.dev_mem(),
                                         conv.desc(), algo,
                                         ws, ws_size,
                                         &beta, y_ref.desc(), y_ref.dev_mem()));
  float y_host_ref[n_y_elem] = {};
  CUXX_CUDA_CHECK(cudaMemcpy(y_host_ref, y_dev_ref, y_size,
                             cudaMemcpyDeviceToHost));

  for (int i = 0; i < n_y_elem; ++i) {
    EXPECT_EQ(y_host_ref[i], y_host[i]) << "Group does not match: " << i;
  }

  CUXX_CUDA_CHECK(cudaFree(y_dev_ref));
  CUXX_CUDA_CHECK(cudaFree(ws));
  CUXX_CUDA_CHECK(cudaFree(y_dev));
  CUXX_CUDA_CHECK(cudaFree(w_dev));
  CUXX_CUDA_CHECK(cudaFree(x_dev));
}

// No value check.
TEST_F(ConvolutionTest, TestGetBackwardDataAlgorithmMaxCount) {
  Convolution<float, float> conv(4, 4, 2, 2, 2, 2, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  int count = conv.GetBackwardDataAlgorithmMaxCount(handle);
  CUXX_UNUSED_VAR(count);
}

// No value check.
TEST_F(ConvolutionTest, TestGetBackwardDataAlgorithm) {
  Convolution<float, float> conv(0, 0, 2, 2, 1, 1, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);

  constexpr int w_k = 4;  // The number of output feature maps.
  constexpr int w_c = 3;  // The number of input feature maps.
  constexpr int w_h = 3;  // The height of each filter.
  constexpr int w_w = 3;  // The width of each filter.
  constexpr int n_w_elem = w_k * w_c * w_h * w_w;
  size_t w_size = sizeof(float) * n_w_elem;
  float w_host[n_w_elem] = {};
  for (int i = 0; i < n_w_elem; ++i) {
    w_host[i] = i * 0.00001;
  }
  float* w_dev = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&w_dev, w_size));
  CUXX_CUDA_CHECK(cudaMemcpy(w_dev, w_host, w_size, cudaMemcpyHostToDevice));
  Filter<float> w(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, w_k, w_c, w_h, w_w,
                  w_dev);

  constexpr int dy_n = 32;
  constexpr int dy_c = 4;
  constexpr int dy_h = 2;
  constexpr int dy_w = 1;
  constexpr int n_dy_elem = dy_n * dy_c * dy_h * dy_w;
  size_t dy_size = sizeof(float) * n_dy_elem;
  float dy_host[n_dy_elem] = {};
  for (int i = 0; i < n_dy_elem; ++i) {
    dy_host[i] = i * 0.0001;
  }
  float* dy_dev = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&dy_dev, dy_size));
  CUXX_CUDA_CHECK(cudaMemcpy(dy_dev, dy_host, dy_size, cudaMemcpyHostToDevice));
  Tensor<float> dy(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dy_n, dy_c, dy_h, dy_w,
                   dy_dev);

  constexpr int dx_n = 32;
  constexpr int dx_c = 3;
  constexpr int dx_h = 6;
  constexpr int dx_w = 4;
  constexpr int n_dx_elem = dx_n * dx_c * dx_h * dx_w;
  size_t dx_size = sizeof(float) * n_dx_elem;
  float* dx_dev = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&dx_dev, dx_size));
  Tensor<float> dx(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dx_n, dx_c, dx_h, dx_w,
                   dx_dev);

  int requested_count = conv.GetBackwardDataAlgorithmMaxCount(handle);
  int returned_count = 0;
  std::vector<cudnnConvolutionBwdDataAlgoPerf_t> results_vec(requested_count);
  conv.GetBackwardDataAlgorithm(handle, w, dy, dx, requested_count,
                                &returned_count, results_vec.data());

  CUXX_CUDA_CHECK(cudaFree(dx_dev));
  CUXX_CUDA_CHECK(cudaFree(dy_dev));
  CUXX_CUDA_CHECK(cudaFree(w_dev));
  EXPECT_EQ(requested_count, returned_count) << "Count does not match.";
}

// No value check.
TEST_F(ConvolutionTest, TestGetBackwardDataWorkspaceSize) {
  Convolution<float, float> conv(0, 0, 2, 2, 1, 1, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);

  constexpr int w_k = 4;  // The number of output feature maps.
  constexpr int w_c = 3;  // The number of input feature maps.
  constexpr int w_h = 3;  // The height of each filter.
  constexpr int w_w = 3;  // The width of each filter.
  constexpr int n_w_elem = w_k * w_c * w_h * w_w;
  size_t w_size = sizeof(float) * n_w_elem;
  float w_host[n_w_elem] = {};
  float* w_dev = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&w_dev, w_size));
  CUXX_CUDA_CHECK(cudaMemcpy(w_dev, w_host, w_size, cudaMemcpyHostToDevice));
  Filter<float> w(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, w_k, w_c, w_h, w_w,
                  w_dev);

  constexpr int dy_n = 32;
  constexpr int dy_c = 4;
  constexpr int dy_h = 2;
  constexpr int dy_w = 1;
  constexpr int n_dy_elem = dy_n * dy_c * dy_h * dy_w;
  size_t dy_size = sizeof(float) * n_dy_elem;
  float dy_host[n_dy_elem] = {};
  float* dy_dev = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&dy_dev, dy_size));
  CUXX_CUDA_CHECK(cudaMemcpy(dy_dev, dy_host, dy_size, cudaMemcpyHostToDevice));
  Tensor<float> dy(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dy_n, dy_c, dy_h, dy_w,
                   dy_dev);

  constexpr int dx_n = 32;
  constexpr int dx_c = 3;
  constexpr int dx_h = 6;
  constexpr int dx_w = 4;
  constexpr int n_dx_elem = dx_n * dx_c * dx_h * dx_w;
  size_t dx_size = sizeof(float) * n_dx_elem;
  float* dx_dev = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&dx_dev, dx_size));
  Tensor<float> dx(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dx_n, dx_c, dx_h, dx_w,
                   dx_dev);

  const size_t size = conv.GetBackwardDataWorkspaceSize(handle, w, dy, dx,
                               CUDNN_CONVOLUTION_BWD_DATA_ALGO_1);
  CUXX_UNUSED_VAR(size);
  CUXX_CUDA_CHECK(cudaFree(dx_dev));
  CUXX_CUDA_CHECK(cudaFree(dy_dev));
  CUXX_CUDA_CHECK(cudaFree(w_dev));
}

// No value check.
TEST_F(ConvolutionTest, TestFindBackwardDataAlgorithm) {
  Convolution<float, float> conv(0, 0, 2, 2, 1, 1, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);

  constexpr int w_k = 4;  // The number of output feature maps.
  constexpr int w_c = 3;  // The number of input feature maps.
  constexpr int w_h = 3;  // The height of each filter.
  constexpr int w_w = 3;  // The width of each filter.
  constexpr int n_w_elem = w_k * w_c * w_h * w_w;
  size_t w_size = sizeof(float) * n_w_elem;
  float w_host[n_w_elem] = {};
  float* w_dev = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&w_dev, w_size));
  CUXX_CUDA_CHECK(cudaMemcpy(w_dev, w_host, w_size, cudaMemcpyHostToDevice));
  Filter<float> w(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, w_k, w_c, w_h, w_w,
                  w_dev);

  constexpr int dy_n = 32;
  constexpr int dy_c = 4;
  constexpr int dy_h = 2;
  constexpr int dy_w = 1;
  constexpr int n_dy_elem = dy_n * dy_c * dy_h * dy_w;
  size_t dy_size = sizeof(float) * n_dy_elem;
  float dy_host[n_dy_elem] = {};
  float* dy_dev = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&dy_dev, dy_size));
  CUXX_CUDA_CHECK(cudaMemcpy(dy_dev, dy_host, dy_size, cudaMemcpyHostToDevice));
  Tensor<float> dy(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dy_n, dy_c, dy_h, dy_w,
                   dy_dev);

  constexpr int dx_n = 32;
  constexpr int dx_c = 3;
  constexpr int dx_h = 6;
  constexpr int dx_w = 4;
  constexpr int n_dx_elem = dx_n * dx_c * dx_h * dx_w;
  size_t dx_size = sizeof(float) * n_dx_elem;
  float* dx_dev = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&dx_dev, dx_size));
  Tensor<float> dx(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dx_n, dx_c, dx_h, dx_w,
                   dx_dev);

  const size_t ws_size = conv.GetBackwardDataWorkspaceSize(handle, w, dy, dx,
                               CUDNN_CONVOLUTION_BWD_DATA_ALGO_1);
  constexpr int requested_algo_count = 6;
  int returned_algo_count = 0;
  cudnnConvolutionBwdDataAlgoPerf_t results[requested_algo_count];
  void* ws = nullptr;
  CUXX_CUDA_CHECK(cudaMalloc(&ws, ws_size));
  conv.FindBackwardDataAlgorithm(handle, w, dy, dx, requested_algo_count,
                                 &returned_algo_count, results, ws, ws_size);
  CUXX_CUDA_CHECK(cudaFree(ws));
  CUXX_CUDA_CHECK(cudaFree(dx_dev));
  CUXX_CUDA_CHECK(cudaFree(dy_dev));
  CUXX_CUDA_CHECK(cudaFree(w_dev));
}

}  // namespace dnn
}  // namespace cuxx
