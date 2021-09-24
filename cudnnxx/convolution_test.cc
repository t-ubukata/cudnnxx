#include "cudnnxx/convolution.h"

#include "gtest/gtest.h"

namespace cudnnxx {

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
  EXPECT_EQ(count, conv.GetGroupCount());
}

TEST_F(ConvolutionTest, TestMathType) {
  Convolution<float, float> conv(4, 4, 2, 2, 2, 2, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  constexpr auto type = CUDNN_DEFAULT_MATH;
  conv.SetMathType(type);
  EXPECT_EQ(type, conv.GetMathType());
}

// No value check.
TEST_F(ConvolutionTest, TestGetForwardAlgorithmMaxCount) {
  Convolution<float, float> conv(4, 4, 2, 2, 2, 2, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  conv.GetForwardAlgorithmMaxCount(handle);
}

TEST_F(ConvolutionTest, TestGetForwardAlgorithm) {
  Convolution<float, float> conv(0, 0, 1, 2, 1, 1, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, 3, 6, 4, nullptr);
  Filter<float> w(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, 3, 3, 3, nullptr);
  Tensor<float> y(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, 4, 4, 1, nullptr);
  int requested_count = conv.GetForwardAlgorithmMaxCount(handle);
  int returned_count = 0;
  std::vector<cudnnConvolutionFwdAlgoPerf_t> results_vec(requested_count);
  conv.GetForwardAlgorithm(handle, x, w, y, requested_count, &returned_count,
                           results_vec.data());
  EXPECT_EQ(requested_count, returned_count);
}

// No value check.
TEST_F(ConvolutionTest, TestGetForwardWorkspaceSize) {
  Convolution<float, float> conv(0, 0, 1, 2, 1, 1, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, 3, 6, 4, nullptr);
  Filter<float> w(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, 3, 3, 3, nullptr);
  Tensor<float> y(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, 4, 4, 1, nullptr);
  conv.GetForwardWorkspaceSize(handle, x, w, y,
                               CUDNN_CONVOLUTION_FWD_ALGO_GEMM);
}

// No value check.
TEST_F(ConvolutionTest, TestFindForwardAlgorithm) {
  Convolution<float, float> conv(0, 0, 1, 2, 1, 1, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  constexpr int x_n = 32;
  constexpr int x_c = 3;
  constexpr int x_h = 6;
  constexpr int x_w = 4;
  constexpr int x_n_elem = x_n * x_c * x_h * x_w;
  size_t x_n_bytes = sizeof(float) * x_n_elem;
  float x_host[x_n_elem] = {};
  float* x_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, x_n_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(x_dev, x_host, x_n_bytes, cudaMemcpyHostToDevice));
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, x_n, x_c, x_h, x_w,
                  x_dev);

  constexpr int w_k = 4;  // The number of output feature maps.
  constexpr int w_c = 3;  // The number of input feature maps.
  constexpr int w_h = 3;  // The height of each filter.
  constexpr int w_w = 3;  // The width of each filter.
  constexpr int w_n_elem = w_k * w_c * w_h * w_w;
  size_t w_n_bytes = sizeof(float) * w_n_elem;
  float w_host[w_n_elem] = {};
  float* w_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&w_dev, w_n_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(w_dev, w_host, w_n_bytes, cudaMemcpyHostToDevice));
  Filter<float> w(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, w_k, w_c, w_h, w_w,
                  w_dev);

  constexpr int y_n = 32;
  constexpr int y_c = 4;
  constexpr int y_h = 4;
  constexpr int y_w = 1;
  constexpr int y_n_elem = y_n * y_c * y_h * y_w;
  size_t y_n_bytes = sizeof(float) * y_n_elem;
  float* y_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_dev, y_n_bytes));
  Tensor<float> y(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, y_n, y_c, y_h, y_w,
                  y_dev);

  size_t ws_n_bytes = conv.GetForwardWorkspaceSize(
      handle, x, w, y, CUDNN_CONVOLUTION_FWD_ALGO_GEMM);
  constexpr int requested_algo_count = 8;
  int returned_algo_count = 0;
  cudnnConvolutionFwdAlgoPerf_t results[requested_algo_count];
  void* ws = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&ws, ws_n_bytes));
  conv.FindForwardAlgorithm(handle, x, w, y, requested_algo_count,
                            &returned_algo_count, results, ws, ws_n_bytes);

  CUDNNXX_CUDA_CHECK(cudaFree(ws));
  CUDNNXX_CUDA_CHECK(cudaFree(y_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(w_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
}

TEST_F(ConvolutionTest, TestGet2dForwardOutputDim) {
  Convolution<float, float> conv(0, 0, 1, 2, 1, 1, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, 3, 6, 4, nullptr);
  Filter<float> f(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, 3, 3, 3, nullptr);
  int n = 0;
  int c = 0;
  int h = 0;
  int w = 0;
  conv.Get2dForwardOutputDim(x, f, &n, &c, &h, &w);
  EXPECT_EQ(32, n);
  EXPECT_EQ(4, c);
  EXPECT_EQ(4, h);
  EXPECT_EQ(1, w);
}

// TODO: Test Nd.

TEST_F(ConvolutionTest, TestForward) {
  Convolution<float, float> conv(0, 0, 1, 2, 1, 1, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  constexpr int x_n = 32;
  constexpr int x_c = 3;
  constexpr int x_h = 6;
  constexpr int x_w = 4;
  constexpr int x_n_elem = x_n * x_c * x_h * x_w;
  size_t x_n_bytes = sizeof(float) * x_n_elem;
  float x_host[x_n_elem] = {};
  for (int i = 0; i < x_n_elem; ++i) {
    x_host[i] = i * 0.0001;
  }
  float* x_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, x_n_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(x_dev, x_host, x_n_bytes, cudaMemcpyHostToDevice));
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, x_n, x_c, x_h, x_w,
                  x_dev);

  constexpr int w_k = 4;  // The number of output feature maps.
  constexpr int w_c = 3;  // The number of input feature maps.
  constexpr int w_h = 3;  // The height of each filter.
  constexpr int w_w = 3;  // The width of each filter.
  constexpr int w_n_elem = w_k * w_c * w_h * w_w;
  size_t w_n_bytes = sizeof(float) * w_n_elem;
  float w_host[w_n_elem] = {};
  for (int i = 0; i < w_n_elem; ++i) {
    w_host[i] = i * 0.00001;
  }
  float* w_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&w_dev, w_n_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(w_dev, w_host, w_n_bytes, cudaMemcpyHostToDevice));
  Filter<float> w(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, w_k, w_c, w_h, w_w,
                  w_dev);

  constexpr int y_n = 32;
  constexpr int y_c = 4;
  constexpr int y_h = 4;
  constexpr int y_w = 1;
  constexpr int y_n_elem = y_n * y_c * y_h * y_w;
  size_t y_n_bytes = sizeof(float) * y_n_elem;
  float* y_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_dev, y_n_bytes));
  Tensor<float> y(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, y_n, y_c, y_h, y_w,
                  y_dev);

  auto algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
  size_t ws_n_bytes = conv.GetForwardWorkspaceSize(handle, x, w, y, algo);
  void* ws = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&ws, ws_n_bytes));

  constexpr float alpha = 1;
  constexpr float beta = 0;
  conv.Forward(handle, alpha, x, w, algo, ws, ws_n_bytes, beta, &y);
  float y_host[y_n_elem] = {};
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(y_host, y_dev, y_n_bytes, cudaMemcpyDeviceToHost));

  float* y_dev_ref = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_dev_ref, y_n_bytes));
  Tensor<float> y_ref(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, y_n, y_c, y_h, y_w,
                      y_dev_ref);
  CUDNNXX_DNN_CHECK(cudnnConvolutionForward(
      handle.raw_handle(), &alpha, x.desc(), x.dev_mem(), w.desc(), w.dev_mem(),
      conv.desc(), algo, ws, ws_n_bytes, &beta, y_ref.desc(), y_ref.dev_mem()));
  float y_host_ref[y_n_elem] = {};
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(y_host_ref, y_dev_ref, y_n_bytes, cudaMemcpyDeviceToHost));

  for (int i = 0; i < y_n_elem; ++i) {
    EXPECT_EQ(y_host_ref[i], y_host[i]) << "at index " << i;
  }

  CUDNNXX_CUDA_CHECK(cudaFree(y_dev_ref));
  CUDNNXX_CUDA_CHECK(cudaFree(ws));
  CUDNNXX_CUDA_CHECK(cudaFree(y_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(w_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
}

// No value check.
TEST_F(ConvolutionTest, TestGetBackwardDataAlgorithmMaxCount) {
  Convolution<float, float> conv(4, 4, 2, 2, 2, 2, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  conv.GetBackwardDataAlgorithmMaxCount(handle);
}

// No value check.
TEST_F(ConvolutionTest, TestGetBackwardDataAlgorithm) {
  Convolution<float, float> conv(0, 0, 2, 2, 1, 1, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  Filter<float> w(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, 3, 3, 3, nullptr);
  Tensor<float> dy(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, 4, 2, 1, nullptr);
  Tensor<float> dx(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, 3, 6, 4, nullptr);
  int requested_count = conv.GetBackwardDataAlgorithmMaxCount(handle);
  int returned_count = 0;
  std::vector<cudnnConvolutionBwdDataAlgoPerf_t> results_vec(requested_count);
  conv.GetBackwardDataAlgorithm(handle, w, dy, dx, requested_count,
                                &returned_count, results_vec.data());
  EXPECT_EQ(requested_count, returned_count);
}

// No value check.
TEST_F(ConvolutionTest, TestGetBackwardDataWorkspaceSize) {
  Convolution<float, float> conv(0, 0, 2, 2, 1, 1, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  Filter<float> w(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, 3, 3, 3, nullptr);
  Tensor<float> dy(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, 4, 2, 1, nullptr);
  Tensor<float> dx(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, 3, 6, 4, nullptr);
  conv.GetBackwardDataWorkspaceSize(handle, w, dy, dx,
                                    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1);
}

// No value check.
TEST_F(ConvolutionTest, TestFindBackwardDataAlgorithm) {
  Convolution<float, float> conv(0, 0, 2, 2, 1, 1, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);

  constexpr int w_k = 4;  // The number of output feature maps.
  constexpr int w_c = 3;  // The number of input feature maps.
  constexpr int w_h = 3;  // The height of each filter.
  constexpr int w_w = 3;  // The width of each filter.
  constexpr int w_n_elem = w_k * w_c * w_h * w_w;
  size_t w_n_bytes = sizeof(float) * w_n_elem;
  float w_host[w_n_elem] = {};
  for (int i = 0; i < w_n_elem; ++i) {
    w_host[i] = i * 0.00001;
  }
  float* w_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&w_dev, w_n_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(w_dev, w_host, w_n_bytes, cudaMemcpyHostToDevice));
  Filter<float> w(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, w_k, w_c, w_h, w_w,
                  w_dev);

  constexpr int dy_n = 32;
  constexpr int dy_c = 4;
  constexpr int dy_h = 2;
  constexpr int dy_w = 1;
  constexpr int dy_n_elem = dy_n * dy_c * dy_h * dy_w;
  size_t dy_n_bytes = sizeof(float) * dy_n_elem;
  float dy_host[dy_n_elem] = {};
  for (int i = 0; i < dy_n_elem; ++i) {
    dy_host[i] = i * 0.0001;
  }
  float* dy_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dy_dev, dy_n_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(dy_dev, dy_host, dy_n_bytes, cudaMemcpyHostToDevice));
  Tensor<float> dy(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dy_n, dy_c, dy_h, dy_w,
                   dy_dev);

  constexpr int dx_n = 32;
  constexpr int dx_c = 3;
  constexpr int dx_h = 6;
  constexpr int dx_w = 4;
  constexpr int dx_n_elem = dx_n * dx_c * dx_h * dx_w;
  size_t dx_n_bytes = sizeof(float) * dx_n_elem;
  float* dx_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dx_dev, dx_n_bytes));
  Tensor<float> dx(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dx_n, dx_c, dx_h, dx_w,
                   dx_dev);

  size_t ws_n_bytes = conv.GetBackwardDataWorkspaceSize(
      handle, w, dy, dx, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1);
  constexpr int requested_algo_count = 6;
  int returned_algo_count = 0;
  cudnnConvolutionBwdDataAlgoPerf_t results[requested_algo_count];
  void* ws = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&ws, ws_n_bytes));
  conv.FindBackwardDataAlgorithm(handle, w, dy, dx, requested_algo_count,
                                 &returned_algo_count, results, ws, ws_n_bytes);
  CUDNNXX_CUDA_CHECK(cudaFree(ws));
  CUDNNXX_CUDA_CHECK(cudaFree(dx_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dy_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(w_dev));
}

TEST_F(ConvolutionTest, TestBackwardData) {
  Convolution<float, float> conv(0, 0, 2, 2, 1, 1, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);

  constexpr int w_k = 4;  // The number of output feature maps.
  constexpr int w_c = 3;  // The number of input feature maps.
  constexpr int w_h = 3;  // The height of each filter.
  constexpr int w_w = 3;  // The width of each filter.
  constexpr int w_n_elem = w_k * w_c * w_h * w_w;
  size_t w_n_bytes = sizeof(float) * w_n_elem;
  float w_host[w_n_elem] = {};
  for (int i = 0; i < w_n_elem; ++i) {
    w_host[i] = i * 0.00001;
  }
  float* w_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&w_dev, w_n_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(w_dev, w_host, w_n_bytes, cudaMemcpyHostToDevice));
  Filter<float> w(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, w_k, w_c, w_h, w_w,
                  w_dev);

  constexpr int dy_n = 32;
  constexpr int dy_c = 4;
  constexpr int dy_h = 2;
  constexpr int dy_w = 1;
  constexpr int dy_n_elem = dy_n * dy_c * dy_h * dy_w;
  size_t dy_n_bytes = sizeof(float) * dy_n_elem;
  float dy_host[dy_n_elem] = {};
  for (int i = 0; i < dy_n_elem; ++i) {
    dy_host[i] = i * 0.0001;
  }
  float* dy_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dy_dev, dy_n_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(dy_dev, dy_host, dy_n_bytes, cudaMemcpyHostToDevice));
  Tensor<float> dy(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dy_n, dy_c, dy_h, dy_w,
                   dy_dev);

  constexpr int dx_n = 32;
  constexpr int dx_c = 3;
  constexpr int dx_h = 6;
  constexpr int dx_w = 4;
  constexpr int dx_n_elem = dx_n * dx_c * dx_h * dx_w;
  size_t dx_n_bytes = sizeof(float) * dx_n_elem;
  float* dx_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dx_dev, dx_n_bytes));
  Tensor<float> dx(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dx_n, dx_c, dx_h, dx_w,
                   dx_dev);

  auto algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  size_t ws_n_bytes =
      conv.GetBackwardDataWorkspaceSize(handle, w, dy, dx, algo);
  void* ws = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&ws, ws_n_bytes));

  float alpha = 1;
  float beta = 0;
  conv.BackwardData(handle, alpha, w, dy, algo, ws, ws_n_bytes, beta, &dx);
  float dx_host[dx_n_elem] = {};
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(dx_host, dx_dev, dx_n_bytes, cudaMemcpyDeviceToHost));

  float* dx_dev_ref = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dx_dev_ref, dx_n_bytes));
  Tensor<float> dx_ref(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dx_n, dx_c, dx_h,
                       dx_w, dx_dev_ref);
  CUDNNXX_DNN_CHECK(cudnnConvolutionBackwardData(
      handle.raw_handle(), &alpha, w.desc(), w.dev_mem(), dy.desc(),
      dy.dev_mem(), conv.desc(), algo, ws, ws_n_bytes, &beta, dx_ref.desc(),
      dx_ref.dev_mem()));
  float dx_host_ref[dx_n_elem] = {};
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(dx_host_ref, dx_dev_ref, dx_n_bytes, cudaMemcpyDeviceToHost));

  for (int i = 0; i < dx_n_elem; ++i) {
    EXPECT_EQ(dx_host_ref[i], dx_host[i]) << "at index " << i;
  }

  CUDNNXX_CUDA_CHECK(cudaFree(dx_dev_ref));
  CUDNNXX_CUDA_CHECK(cudaFree(ws));
  CUDNNXX_CUDA_CHECK(cudaFree(dx_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dy_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(w_dev));
}

// No value check.
TEST_F(ConvolutionTest, TestGetBackwardFilterAlgorithmMaxCount) {
  Convolution<float, float> conv(4, 4, 2, 2, 2, 2, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  conv.GetBackwardFilterAlgorithmMaxCount(handle);
}

// No value check.
TEST_F(ConvolutionTest, TestGetBackwardFilterAlgorithm) {
  Convolution<float, float> conv(0, 0, 2, 2, 1, 1, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, 3, 6, 4, nullptr);
  Tensor<float> dy(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, 4, 2, 1, nullptr);
  Filter<float> dw(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, 3, 3, 3, nullptr);
  int requested_count = conv.GetBackwardFilterAlgorithmMaxCount(handle);
  int returned_count = 0;
  std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> results_vec(requested_count);
  conv.GetBackwardFilterAlgorithm(handle, x, dy, dw, requested_count,
                                  &returned_count, results_vec.data());
  EXPECT_EQ(requested_count, returned_count);
}

// No value check.
TEST_F(ConvolutionTest, TestGetBackwardFilterWorkspaceSize) {
  Convolution<float, float> conv(0, 0, 2, 2, 1, 1, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, 3, 6, 4, nullptr);
  Tensor<float> dy(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, 4, 2, 1, nullptr);
  Filter<float> dw(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, 3, 3, 3, nullptr);
  conv.GetBackwardFilterWorkspaceSize(handle, x, dy, dw,
                                      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1);
}

// No value check.
TEST_F(ConvolutionTest, TestFindBackwardFilterAlgorithm) {
  Convolution<float, float> conv(0, 0, 2, 2, 1, 1, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);

  constexpr int x_n = 32;
  constexpr int x_c = 3;
  constexpr int x_h = 6;
  constexpr int x_w = 4;
  constexpr int x_n_elem = x_n * x_c * x_h * x_w;
  size_t x_n_bytes = sizeof(float) * x_n_elem;
  float x_host[x_n_elem] = {};
  for (int i = 0; i < x_n_elem; ++i) {
    x_host[i] = i * 0.00001;
  }
  float* x_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, x_n_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(x_dev, x_host, x_n_bytes, cudaMemcpyHostToDevice));
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, x_n, x_c, x_h, x_w,
                  x_dev);

  constexpr int dy_n = 32;
  constexpr int dy_c = 4;
  constexpr int dy_h = 2;
  constexpr int dy_w = 1;
  constexpr int dy_n_elem = dy_n * dy_c * dy_h * dy_w;
  size_t dy_n_bytes = sizeof(float) * dy_n_elem;
  float dy_host[dy_n_elem] = {};
  for (int i = 0; i < dy_n_elem; ++i) {
    dy_host[i] = i * 0.0001;
  }
  float* dy_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dy_dev, dy_n_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(dy_dev, dy_host, dy_n_bytes, cudaMemcpyHostToDevice));
  Tensor<float> dy(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dy_n, dy_c, dy_h, dy_w,
                   dy_dev);

  constexpr int dw_k = 4;  // The number of output feature maps.
  constexpr int dw_c = 3;  // The number of input feature maps.
  constexpr int dw_h = 3;  // The height of each filter.
  constexpr int dw_w = 3;  // The width of each filter.
  constexpr int dw_n_elem = dw_k * dw_c * dw_h * dw_w;
  size_t dw_n_bytes = sizeof(float) * dw_n_elem;
  float* dw_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dw_dev, dw_n_bytes));
  Filter<float> dw(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dw_k, dw_c, dw_h, dw_w,
                   dw_dev);

  size_t ws_n_bytes = conv.GetBackwardFilterWorkspaceSize(
      handle, x, dy, dw, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1);
  constexpr int requested_algo_count = 6;
  int returned_algo_count = 0;
  cudnnConvolutionBwdFilterAlgoPerf_t results[requested_algo_count];
  void* ws = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&ws, ws_n_bytes));
  conv.FindBackwardFilterAlgorithm(handle, x, dy, dw, requested_algo_count,
                                   &returned_algo_count, results, ws,
                                   ws_n_bytes);
  CUDNNXX_CUDA_CHECK(cudaFree(ws));
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dy_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dw_dev));
}

TEST_F(ConvolutionTest, TestBackwardFilter) {
  Convolution<float, float> conv(0, 0, 2, 2, 1, 1, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  constexpr int x_n = 32;
  constexpr int x_c = 3;
  constexpr int x_h = 6;
  constexpr int x_w = 4;
  constexpr int x_n_elem = x_n * x_c * x_h * x_w;
  size_t x_n_bytes = sizeof(float) * x_n_elem;
  float x_host[x_n_elem] = {};
  for (int i = 0; i < x_n_elem; ++i) {
    x_host[i] = i * 0.0001;
  }
  float* x_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, x_n_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(x_dev, x_host, x_n_bytes, cudaMemcpyHostToDevice));
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, x_n, x_c, x_h, x_w,
                  x_dev);

  constexpr int dy_n = 32;
  constexpr int dy_c = 4;
  constexpr int dy_h = 2;
  constexpr int dy_w = 1;
  constexpr int dy_n_elem = dy_n * dy_c * dy_h * dy_w;
  size_t dy_n_bytes = sizeof(float) * dy_n_elem;
  float dy_host[dy_n_elem] = {};
  for (int i = 0; i < dy_n_elem; ++i) {
    dy_host[i] = i * 0.00001;
  }
  float* dy_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dy_dev, dy_n_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(dy_dev, dy_host, dy_n_bytes, cudaMemcpyHostToDevice));
  Tensor<float> dy(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dy_n, dy_c, dy_h, dy_w,
                   dy_dev);

  constexpr int dw_k = 4;  // The number of output feature maps.
  constexpr int dw_c = 3;  // The number of input feature maps.
  constexpr int dw_h = 3;  // The height of each filter.
  constexpr int dw_w = 3;  // The width of each filter.
  constexpr int dw_n_elem = dw_k * dw_c * dw_h * dw_w;
  size_t dw_n_bytes = sizeof(float) * dw_n_elem;
  float* dw_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dw_dev, dw_n_bytes));
  Filter<float> dw(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dw_k, dw_c, dw_h, dw_w,
                   dw_dev);

  auto algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  size_t ws_n_bytes =
      conv.GetBackwardFilterWorkspaceSize(handle, x, dy, dw, algo);
  void* ws = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&ws, ws_n_bytes));

  float alpha = 1;
  float beta = 0;
  conv.BackwardFilter(handle, alpha, x, dy, algo, ws, ws_n_bytes, beta, &dw);
  float dw_host[dw_n_elem] = {};
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(dw_host, dw_dev, dw_n_bytes, cudaMemcpyDeviceToHost));

  float* dw_dev_ref = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dw_dev_ref, dw_n_bytes));
  Filter<float> dw_ref(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dw_k, dw_c, dw_h,
                       dw_w, dw_dev_ref);
  CUDNNXX_DNN_CHECK(cudnnConvolutionBackwardFilter(
      handle.raw_handle(), &alpha, x.desc(), x.dev_mem(), dy.desc(),
      dy.dev_mem(), conv.desc(), algo, ws, ws_n_bytes, &beta, dw_ref.desc(),
      dw_ref.dev_mem()));
  float dw_host_ref[dw_n_elem] = {};
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(dw_host_ref, dw_dev_ref, dw_n_bytes, cudaMemcpyDeviceToHost));

  for (int i = 0; i < dw_n_elem; ++i) {
    EXPECT_EQ(dw_host_ref[i], dw_host[i]) << "at index " << i;
  }

  CUDNNXX_CUDA_CHECK(cudaFree(dw_dev_ref));
  CUDNNXX_CUDA_CHECK(cudaFree(ws));
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dy_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dw_dev));
}

}  // namespace cudnnxx
