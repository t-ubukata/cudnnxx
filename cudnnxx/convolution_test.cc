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
  EXPECT_EQ(count, conv.GetGroupCount()) << "Group count does not match.";
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
  CUDNNXX_UNUSED_VAR(count);
}

// No value check.
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
  EXPECT_EQ(requested_count, returned_count) << "Count does not match.";
}

// No value check.
TEST_F(ConvolutionTest, TestGetForwardWorkspaceSize) {
  Convolution<float, float> conv(0, 0, 1, 2, 1, 1, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, 3, 6, 4, nullptr);
  Filter<float> w(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, 3, 3, 3, nullptr);
  Tensor<float> y(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, 4, 4, 1, nullptr);
  size_t size = conv.GetForwardWorkspaceSize(handle, x, w, y,
                                             CUDNN_CONVOLUTION_FWD_ALGO_GEMM);
  CUDNNXX_UNUSED_VAR(size);
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
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, x_size));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host, x_size, cudaMemcpyHostToDevice));
  Tensor<float> x_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, x_n, x_c, x_h,
                         x_w, x_dev);

  constexpr int w_k = 4;  // The number of output feature maps.
  constexpr int w_c = 3;  // The number of input feature maps.
  constexpr int w_h = 3;  // The height of each filter.
  constexpr int w_w = 3;  // The width of each filter.
  constexpr int n_w_elem = w_k * w_c * w_h * w_w;
  size_t w_size = sizeof(float) * n_w_elem;
  float w_host[n_w_elem] = {};
  float* w_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&w_dev, w_size));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(w_dev, w_host, w_size, cudaMemcpyHostToDevice));
  Filter<float> w_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, w_k, w_c, w_h,
                         w_w, w_dev);

  constexpr int y_n = 32;
  constexpr int y_c = 4;
  constexpr int y_h = 4;
  constexpr int y_w = 1;
  constexpr int n_y_elem = y_n * y_c * y_h * y_w;
  size_t y_size = sizeof(float) * n_y_elem;
  float* y_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_dev, y_size));
  Tensor<float> y_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, y_n, y_c, y_h,
                         y_w, y_dev);

  size_t ws_size = conv.GetForwardWorkspaceSize(
      handle, x_tensor, w_tensor, y_tensor, CUDNN_CONVOLUTION_FWD_ALGO_GEMM);
  constexpr int requested_algo_count = 8;
  int returned_algo_count = 0;
  cudnnConvolutionFwdAlgoPerf_t results[requested_algo_count];
  void* ws = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&ws, ws_size));
  conv.FindForwardAlgorithm(handle, x_tensor, w_tensor, y_tensor,
                            requested_algo_count, &returned_algo_count, results,
                            ws, ws_size);

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
  EXPECT_EQ(32, n) << "n does not match.";
  EXPECT_EQ(4, c) << "c does not match.";
  EXPECT_EQ(4, h) << "h does not match.";
  EXPECT_EQ(1, w) << "w does not match.";
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
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, x_size));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host, x_size, cudaMemcpyHostToDevice));
  Tensor<float> x_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, x_n, x_c, x_h,
                         x_w, x_dev);

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
  CUDNNXX_CUDA_CHECK(cudaMalloc(&w_dev, w_size));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(w_dev, w_host, w_size, cudaMemcpyHostToDevice));
  Filter<float> w_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, w_k, w_c, w_h,
                         w_w, w_dev);

  constexpr int y_n = 32;
  constexpr int y_c = 4;
  constexpr int y_h = 4;
  constexpr int y_w = 1;
  constexpr int n_y_elem = y_n * y_c * y_h * y_w;
  size_t y_size = sizeof(float) * n_y_elem;
  float* y_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_dev, y_size));
  Tensor<float> y_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, y_n, y_c, y_h,
                         y_w, y_dev);

  auto algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
  size_t ws_size =
      conv.GetForwardWorkspaceSize(handle, x_tensor, w_tensor, y_tensor, algo);
  void* ws = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&ws, ws_size));

  constexpr float alpha = 1;
  constexpr float beta = 0;
  conv.Forward(handle, alpha, x_tensor, w_tensor, algo, ws, ws_size, beta,
               &y_tensor);
  float y_host[n_y_elem] = {};
  CUDNNXX_CUDA_CHECK(cudaMemcpy(y_host, y_dev, y_size, cudaMemcpyDeviceToHost));

  float* y_dev_ref = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_dev_ref, y_size));
  Tensor<float> y_tensor_ref(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, y_n, y_c, y_h,
                             y_w, y_dev_ref);
  CUDNNXX_DNN_CHECK(cudnnConvolutionForward(
      handle.raw_handle(), &alpha, x_tensor.desc(), x_tensor.dev_mem(),
      w_tensor.desc(), w_tensor.dev_mem(), conv.desc(), algo, ws, ws_size,
      &beta, y_tensor_ref.desc(), y_tensor_ref.dev_mem()));
  float y_host_ref[n_y_elem] = {};
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(y_host_ref, y_dev_ref, y_size, cudaMemcpyDeviceToHost));

  for (int i = 0; i < n_y_elem; ++i) {
    EXPECT_EQ(y_host_ref[i], y_host[i]) << "Value does not match: " << i;
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
  int count = conv.GetBackwardDataAlgorithmMaxCount(handle);
  CUDNNXX_UNUSED_VAR(count);
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
  EXPECT_EQ(requested_count, returned_count) << "Count does not match.";
}

// No value check.
TEST_F(ConvolutionTest, TestGetBackwardDataWorkspaceSize) {
  Convolution<float, float> conv(0, 0, 2, 2, 1, 1, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  Filter<float> w(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, 3, 3, 3, nullptr);
  Tensor<float> dy(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, 4, 2, 1, nullptr);
  Tensor<float> dx(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, 3, 6, 4, nullptr);
  size_t size = conv.GetBackwardDataWorkspaceSize(
      handle, w, dy, dx, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1);
  CUDNNXX_UNUSED_VAR(size);
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
  for (int i = 0; i < n_w_elem; ++i) {
    w_host[i] = i * 0.00001;
  }
  float* w_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&w_dev, w_size));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(w_dev, w_host, w_size, cudaMemcpyHostToDevice));
  Filter<float> w_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, w_k, w_c, w_h,
                         w_w, w_dev);

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
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dy_dev, dy_size));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(dy_dev, dy_host, dy_size, cudaMemcpyHostToDevice));
  Tensor<float> dy_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dy_n, dy_c, dy_h,
                          dy_w, dy_dev);

  constexpr int dx_n = 32;
  constexpr int dx_c = 3;
  constexpr int dx_h = 6;
  constexpr int dx_w = 4;
  constexpr int n_dx_elem = dx_n * dx_c * dx_h * dx_w;
  size_t dx_size = sizeof(float) * n_dx_elem;
  float* dx_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dx_dev, dx_size));
  Tensor<float> dx_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dx_n, dx_c, dx_h,
                          dx_w, dx_dev);

  size_t ws_size =
      conv.GetBackwardDataWorkspaceSize(handle, w_tensor, dy_tensor, dx_tensor,
                                        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1);
  constexpr int requested_algo_count = 6;
  int returned_algo_count = 0;
  cudnnConvolutionBwdDataAlgoPerf_t results[requested_algo_count];
  void* ws = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&ws, ws_size));
  conv.FindBackwardDataAlgorithm(handle, w_tensor, dy_tensor, dx_tensor,
                                 requested_algo_count, &returned_algo_count,
                                 results, ws, ws_size);
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
  constexpr int n_w_elem = w_k * w_c * w_h * w_w;
  size_t w_size = sizeof(float) * n_w_elem;
  float w_host[n_w_elem] = {};
  for (int i = 0; i < n_w_elem; ++i) {
    w_host[i] = i * 0.00001;
  }
  float* w_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&w_dev, w_size));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(w_dev, w_host, w_size, cudaMemcpyHostToDevice));
  Filter<float> w_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, w_k, w_c, w_h,
                         w_w, w_dev);

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
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dy_dev, dy_size));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(dy_dev, dy_host, dy_size, cudaMemcpyHostToDevice));
  Tensor<float> dy_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dy_n, dy_c, dy_h,
                          dy_w, dy_dev);

  constexpr int dx_n = 32;
  constexpr int dx_c = 3;
  constexpr int dx_h = 6;
  constexpr int dx_w = 4;
  constexpr int n_dx_elem = dx_n * dx_c * dx_h * dx_w;
  size_t dx_size = sizeof(float) * n_dx_elem;
  float* dx_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dx_dev, dx_size));
  Tensor<float> dx_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dx_n, dx_c, dx_h,
                          dx_w, dx_dev);

  auto algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  size_t ws_size = conv.GetBackwardDataWorkspaceSize(
      handle, w_tensor, dy_tensor, dx_tensor, algo);
  void* ws = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&ws, ws_size));

  constexpr float alpha = 1;
  constexpr float beta = 0;
  conv.BackwardData(handle, alpha, w_tensor, dy_tensor, algo, ws, ws_size, beta,
                    &dx_tensor);
  float dx_host[n_dx_elem] = {};
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(dx_host, dx_dev, dx_size, cudaMemcpyDeviceToHost));

  float* dx_dev_ref = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dx_dev_ref, dx_size));
  Tensor<float> dx_tensor_ref(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dx_n, dx_c,
                              dx_h, dx_w, dx_dev_ref);
  CUDNNXX_DNN_CHECK(cudnnConvolutionBackwardData(
      handle.raw_handle(), &alpha, w_tensor.desc(), w_tensor.dev_mem(),
      dy_tensor.desc(), dy_tensor.dev_mem(), conv.desc(), algo, ws, ws_size,
      &beta, dx_tensor_ref.desc(), dx_tensor_ref.dev_mem()));
  float dx_host_ref[n_dx_elem] = {};
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(dx_host_ref, dx_dev_ref, dx_size, cudaMemcpyDeviceToHost));

  for (int i = 0; i < n_dx_elem; ++i) {
    EXPECT_EQ(dx_host_ref[i], dx_host[i]) << "Value does not match: " << i;
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
  int count = conv.GetBackwardFilterAlgorithmMaxCount(handle);
  CUDNNXX_UNUSED_VAR(count);
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
  EXPECT_EQ(requested_count, returned_count) << "Count does not match.";
}

// No value check.
TEST_F(ConvolutionTest, TestGetBackwardFilterWorkspaceSize) {
  Convolution<float, float> conv(0, 0, 2, 2, 1, 1, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, 3, 6, 4, nullptr);
  Tensor<float> dy(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, 4, 2, 1, nullptr);
  Filter<float> dw(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, 3, 3, 3, nullptr);
  size_t size = conv.GetBackwardFilterWorkspaceSize(
      handle, x, dy, dw, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1);
  CUDNNXX_UNUSED_VAR(size);
}

// No value check.
TEST_F(ConvolutionTest, TestFindBackwardFilterAlgorithm) {
  Convolution<float, float> conv(0, 0, 2, 2, 1, 1, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);

  constexpr int x_n = 32;
  constexpr int x_c = 3;
  constexpr int x_h = 6;
  constexpr int x_w = 4;
  constexpr int n_x_elem = x_n * x_c * x_h * x_w;
  size_t x_size = sizeof(float) * n_x_elem;
  float x_host[n_x_elem] = {};
  for (int i = 0; i < n_x_elem; ++i) {
    x_host[i] = i * 0.00001;
  }
  float* x_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, x_size));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host, x_size, cudaMemcpyHostToDevice));
  Tensor<float> x_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, x_n, x_c, x_h,
                         x_w, x_dev);

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
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dy_dev, dy_size));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(dy_dev, dy_host, dy_size, cudaMemcpyHostToDevice));
  Tensor<float> dy_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dy_n, dy_c, dy_h,
                          dy_w, dy_dev);

  constexpr int dw_k = 4;  // The number of output feature maps.
  constexpr int dw_c = 3;  // The number of input feature maps.
  constexpr int dw_h = 3;  // The height of each filter.
  constexpr int dw_w = 3;  // The width of each filter.
  constexpr int n_dw_elem = dw_k * dw_c * dw_h * dw_w;
  size_t dw_size = sizeof(float) * n_dw_elem;
  float* dw_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dw_dev, dw_size));
  Filter<float> dw_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dw_k, dw_c, dw_h,
                          dw_w, dw_dev);

  size_t ws_size = conv.GetBackwardFilterWorkspaceSize(
      handle, x_tensor, dy_tensor, dw_tensor,
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1);
  constexpr int requested_algo_count = 6;
  int returned_algo_count = 0;
  cudnnConvolutionBwdFilterAlgoPerf_t results[requested_algo_count];
  void* ws = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&ws, ws_size));
  conv.FindBackwardFilterAlgorithm(handle, x_tensor, dy_tensor, dw_tensor,
                                   requested_algo_count, &returned_algo_count,
                                   results, ws, ws_size);
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
  constexpr int n_x_elem = x_n * x_c * x_h * x_w;
  size_t x_size = sizeof(float) * n_x_elem;
  float x_host[n_x_elem] = {};
  for (int i = 0; i < n_x_elem; ++i) {
    x_host[i] = i * 0.0001;
  }
  float* x_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, x_size));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host, x_size, cudaMemcpyHostToDevice));
  Tensor<float> x_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, x_n, x_c, x_h,
                         x_w, x_dev);

  constexpr int dy_n = 32;
  constexpr int dy_c = 4;
  constexpr int dy_h = 2;
  constexpr int dy_w = 1;
  constexpr int n_dy_elem = dy_n * dy_c * dy_h * dy_w;
  size_t dy_size = sizeof(float) * n_dy_elem;
  float dy_host[n_dy_elem] = {};
  for (int i = 0; i < n_dy_elem; ++i) {
    dy_host[i] = i * 0.00001;
  }
  float* dy_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dy_dev, dy_size));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(dy_dev, dy_host, dy_size, cudaMemcpyHostToDevice));
  Tensor<float> dy_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dy_n, dy_c, dy_h,
                          dy_w, dy_dev);

  constexpr int dw_k = 4;  // The number of output feature maps.
  constexpr int dw_c = 3;  // The number of input feature maps.
  constexpr int dw_h = 3;  // The height of each filter.
  constexpr int dw_w = 3;  // The width of each filter.
  constexpr int n_dw_elem = dw_k * dw_c * dw_h * dw_w;
  size_t dw_size = sizeof(float) * n_dw_elem;
  float* dw_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dw_dev, dw_size));
  Filter<float> dw_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dw_k, dw_c, dw_h,
                          dw_w, dw_dev);

  auto algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  size_t ws_size = conv.GetBackwardFilterWorkspaceSize(
      handle, x_tensor, dy_tensor, dw_tensor, algo);
  void* ws = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&ws, ws_size));

  constexpr float alpha = 1;
  constexpr float beta = 0;
  conv.BackwardFilter(handle, alpha, x_tensor, dy_tensor, algo, ws, ws_size,
                      beta, &dw_tensor);
  float dw_host[n_dw_elem] = {};
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(dw_host, dw_dev, dw_size, cudaMemcpyDeviceToHost));

  float* dw_dev_ref = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dw_dev_ref, dw_size));
  Filter<float> dw_tensor_ref(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dw_k, dw_c,
                              dw_h, dw_w, dw_dev_ref);
  CUDNNXX_DNN_CHECK(cudnnConvolutionBackwardFilter(
      handle.raw_handle(), &alpha, x_tensor.desc(), x_tensor.dev_mem(),
      dy_tensor.desc(), dy_tensor.dev_mem(), conv.desc(), algo, ws, ws_size,
      &beta, dw_tensor_ref.desc(), dw_tensor_ref.dev_mem()));
  float dw_host_ref[n_dw_elem] = {};
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(dw_host_ref, dw_dev_ref, dw_size, cudaMemcpyDeviceToHost));

  for (int i = 0; i < n_dw_elem; ++i) {
    EXPECT_EQ(dw_host_ref[i], dw_host[i]) << "Value does not match: " << i;
  }

  CUDNNXX_CUDA_CHECK(cudaFree(dw_dev_ref));
  CUDNNXX_CUDA_CHECK(cudaFree(ws));
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dy_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dw_dev));
}

}  // namespace cudnnxx
