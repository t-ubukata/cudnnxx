#include "cudnnxx/dropout.h"

#include "cudnnxx/activation.h"
#include "gtest/gtest.h"

namespace cudnnxx {

class DropoutTest : public ::testing::Test {
 protected:
  Handle handle;
};

TEST_F(DropoutTest, TestConstructor) {
  float dropout_p = 0.5;
  unsigned long long seed = 20200620;
  Dropout<float> dropout(handle, dropout_p, seed);
}

TEST_F(DropoutTest, TestForward) {
  constexpr int n = 2;
  constexpr int c = 3;
  constexpr int h = 2;
  constexpr int w = 2;
  constexpr int n_elem = n * c * h * w;
  size_t size = sizeof(float) * n_elem;

  // Target.

  float x_host[n_elem] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                          0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                          1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4};
  float* x_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, size));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host, size, cudaMemcpyHostToDevice));
  Tensor<float> x_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w,
                         x_dev);

  float* y_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_dev, size));
  Tensor<float> y_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w,
                         y_dev);

  float dropout_p = 0.5;
  unsigned long long seed = 20200627;
  Dropout<float> dropout(handle, dropout_p, seed);
  dropout.Forward(handle, x_tensor, &y_tensor);

  float y_host[n_elem] = {};
  CUDNNXX_CUDA_CHECK(cudaMemcpy(y_host, y_dev, size, cudaMemcpyDeviceToHost));

  // Reference.

  float* y_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_ref_dev, size));
  Tensor<float> y_ref_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w,
                             y_ref_dev);

  cudnnDropoutDescriptor_t dropout_desc_ref;
  CUDNNXX_DNN_CHECK(cudnnCreateDropoutDescriptor(&dropout_desc_ref));
  void* states_ref = nullptr;
  size_t states_size_in_bytes_ref = 0;
  CUDNNXX_DNN_CHECK(cudnnDropoutGetStatesSize(handle.raw_handle(),
                                              &states_size_in_bytes_ref));
  CUDNNXX_CUDA_CHECK(cudaMalloc(&states_ref, states_size_in_bytes_ref));
  CUDNNXX_DNN_CHECK(cudnnSetDropoutDescriptor(
      dropout_desc_ref, handle.raw_handle(), dropout_p, states_ref,
      states_size_in_bytes_ref, seed));

  size_t reserve_space_size_in_bytes_ref = 0;
  CUDNNXX_DNN_CHECK(cudnnDropoutGetReserveSpaceSize(
      x_tensor.desc(), &reserve_space_size_in_bytes_ref));
  void* reserve_space_ref = nullptr;
  CUDNNXX_CUDA_CHECK(
      cudaMalloc(&reserve_space_ref, reserve_space_size_in_bytes_ref));

  CUDNNXX_DNN_CHECK(cudnnDropoutForward(
      handle.raw_handle(), dropout_desc_ref, x_tensor.desc(),
      x_tensor.dev_mem(), y_ref_tensor.desc(), y_ref_tensor.dev_mem(),
      reserve_space_ref, reserve_space_size_in_bytes_ref));

  float y_ref_host[n_elem] = {};
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(y_ref_host, y_ref_dev, size, cudaMemcpyDeviceToHost));

  // Check.

  for (int i = 0; i < n_elem; ++i) {
    EXPECT_NEAR(y_ref_host[i], y_host[i], 1e-4) << "i: " << i;
  }
  CUDNNXX_CUDA_CHECK(cudaFree(reserve_space_ref));
  CUDNNXX_CUDA_CHECK(cudaFree(states_ref));
  CUDNNXX_CUDA_CHECK(cudaFree(y_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(y_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
  CUDNNXX_DNN_CHECK(cudnnDestroyDropoutDescriptor(dropout_desc_ref));
}

TEST_F(DropoutTest, TestBackward) {
  constexpr int n = 2;
  constexpr int c = 3;
  constexpr int h = 2;
  constexpr int w = 2;
  constexpr int n_elem = n * c * h * w;
  size_t size = sizeof(float) * n_elem;

  // Target.

  float x_host[n_elem] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                          0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                          1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4};
  float* x_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, size));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host, size, cudaMemcpyHostToDevice));
  Tensor<float> x_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w,
                         x_dev);

  float* y_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_dev, size));
  Tensor<float> y_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w,
                         y_dev);

  float dropout_p = 0.5;
  unsigned long long seed = 20200627;
  Dropout<float> dropout(handle, dropout_p, seed);
  dropout.Forward(handle, x_tensor, &y_tensor);

  float* dx_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dx_dev, size));
  Tensor<float> dx_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w,
                          dx_dev);

  dropout.Backward(handle, y_tensor, &dx_tensor);

  float dx_host[n_elem] = {};
  CUDNNXX_CUDA_CHECK(cudaMemcpy(dx_host, dx_dev, size, cudaMemcpyDeviceToHost));

  // Reference.

  float* y_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_ref_dev, size));
  Tensor<float> y_ref_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w,
                             y_ref_dev);

  cudnnDropoutDescriptor_t dropout_desc_ref;
  CUDNNXX_DNN_CHECK(cudnnCreateDropoutDescriptor(&dropout_desc_ref));
  void* states_ref = nullptr;
  size_t states_size_in_bytes_ref = 0;
  CUDNNXX_DNN_CHECK(cudnnDropoutGetStatesSize(handle.raw_handle(),
                                              &states_size_in_bytes_ref));
  CUDNNXX_CUDA_CHECK(cudaMalloc(&states_ref, states_size_in_bytes_ref));
  CUDNNXX_DNN_CHECK(cudnnSetDropoutDescriptor(
      dropout_desc_ref, handle.raw_handle(), dropout_p, states_ref,
      states_size_in_bytes_ref, seed));

  size_t reserve_space_size_in_bytes_ref = 0;
  CUDNNXX_DNN_CHECK(cudnnDropoutGetReserveSpaceSize(
      x_tensor.desc(), &reserve_space_size_in_bytes_ref));
  void* reserve_space_ref = nullptr;
  CUDNNXX_CUDA_CHECK(
      cudaMalloc(&reserve_space_ref, reserve_space_size_in_bytes_ref));

  CUDNNXX_DNN_CHECK(cudnnDropoutForward(
      handle.raw_handle(), dropout_desc_ref, x_tensor.desc(),
      x_tensor.dev_mem(), y_ref_tensor.desc(), y_ref_tensor.dev_mem(),
      reserve_space_ref, reserve_space_size_in_bytes_ref));

  float* dx_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dx_ref_dev, size));
  Tensor<float> dx_ref_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w,
                              dx_ref_dev);

  CUDNNXX_DNN_CHECK(cudnnDropoutBackward(
      handle.raw_handle(), dropout_desc_ref, y_ref_tensor.desc(),
      y_ref_tensor.dev_mem(), dx_ref_tensor.desc(), dx_ref_tensor.dev_mem(),
      reserve_space_ref, reserve_space_size_in_bytes_ref));

  float dx_ref_host[n_elem] = {};
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(dx_ref_host, dx_ref_dev, size, cudaMemcpyDeviceToHost));

  // Check.

  for (int i = 0; i < n_elem; ++i) {
    EXPECT_NEAR(dx_ref_host[i], dx_host[i], 1e-4) << "i: " << i;
  }
  CUDNNXX_CUDA_CHECK(cudaFree(dx_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(reserve_space_ref));
  CUDNNXX_CUDA_CHECK(cudaFree(states_ref));
  CUDNNXX_CUDA_CHECK(cudaFree(y_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dx_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(y_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
  CUDNNXX_DNN_CHECK(cudnnDestroyDropoutDescriptor(dropout_desc_ref));
}

}  // namespace cudnnxx
