#include "cudnnxx/cudnnxx.h"
#include "gtest/gtest.h"

namespace cudnnxx {

class ExampleTest : public ::testing::Test {
 protected:
  Handle handle;
};

TEST_F(ExampleTest, TestConvolutionForward2d) {
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
  CUDNNXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host, x_n_bytes, cudaMemcpyHostToDevice));
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
  CUDNNXX_CUDA_CHECK(cudaMemcpy(w_dev, w_host, w_n_bytes, cudaMemcpyHostToDevice));
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
  CUDNNXX_CUDA_CHECK(cudaMemcpy(y_host, y_dev, y_n_bytes, cudaMemcpyDeviceToHost));

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
    EXPECT_EQ(y_host_ref[i], y_host[i]) << "Value does not match: " << i;
  }

  CUDNNXX_CUDA_CHECK(cudaFree(y_dev_ref));
  CUDNNXX_CUDA_CHECK(cudaFree(ws));
  CUDNNXX_CUDA_CHECK(cudaFree(y_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(w_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
}

TEST_F(ExampleTest, TestcudnnConvolutionForward2d) {
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
  CUDNNXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host, x_n_bytes, cudaMemcpyHostToDevice));
  cudnnTensorDescriptor_t x_desc;
  CUDNNXX_DNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
  CUDNNXX_DNN_CHECK(cudnnSetTensor4dDescriptor(
      x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, x_n, x_c, x_h, x_w));

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
  CUDNNXX_CUDA_CHECK(cudaMemcpy(w_dev, w_host, w_n_elem, cudaMemcpyHostToDevice));
  Filter<float> w(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, w_k, w_c, w_h, w_w,
                  w_dev);

  cudnnFilterDescriptor_t w_desc;
  CUDNNXX_DNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
  CUDNNXX_DNN_CHECK(cudnnSetFilter4dDescriptor(
      w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, w_k, w_c, w_h, w_w));

  constexpr int y_n = 32;
  constexpr int y_c = 4;
  constexpr int y_h = 4;
  constexpr int y_w = 1;
  constexpr int y_n_elem = y_n * y_c * y_h * y_w;
  size_t y_n_bytes = sizeof(float) * y_n_elem;
  float* y_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_dev, y_n_bytes));
  cudnnTensorDescriptor_t y_desc;
  CUDNNXX_DNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
  CUDNNXX_DNN_CHECK(cudnnSetTensor4dDescriptor(
      y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, y_n, y_c, y_h, y_w));

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNNXX_DNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNNXX_DNN_CHECK(cudnnSetConvolution2dDescriptor(
      conv_desc, 0, 0, 1, 2, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
  cudnnHandle_t raw_handle;
  CUDNNXX_DNN_CHECK(cudnnCreate(&raw_handle));

  auto algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
  size_t ws_n_bytes = 0;
  CUDNNXX_DNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      raw_handle, x_desc, w_desc, conv_desc, y_desc, algo, &ws_n_bytes));
  void* ws = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&ws, ws_n_bytes));

  constexpr float alpha = 1;
  constexpr float beta = 0;
  CUDNNXX_DNN_CHECK(cudnnConvolutionForward(raw_handle, &alpha, x_desc, x_dev,
                                            w_desc, w_dev, conv_desc, algo, ws,
                                            ws_n_bytes, &beta, y_desc, y_dev));

  float y_host[y_n_elem] = {};
  CUDNNXX_CUDA_CHECK(cudaMemcpy(y_host, y_dev, y_n_bytes, cudaMemcpyDeviceToHost));

  CUDNNXX_DNN_CHECK(cudnnDestroy(raw_handle));
  CUDNNXX_DNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNNXX_CUDA_CHECK(cudaFree(ws));
  CUDNNXX_DNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
  CUDNNXX_CUDA_CHECK(cudaFree(y_dev));
  CUDNNXX_DNN_CHECK(cudnnDestroyFilterDescriptor(w_desc));
  CUDNNXX_CUDA_CHECK(cudaFree(w_dev));
  CUDNNXX_DNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
}

}  // namespace cudnnxx
