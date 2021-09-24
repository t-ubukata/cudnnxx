#include "cudnnxx/activation.h"

#include "gtest/gtest.h"

namespace cudnnxx {

class ActivationTest : public ::testing::Test {
 protected:
  Handle handle;
};

TEST_F(ActivationTest, TestConstructor) {
  Activation<float, float> activation(CUDNN_ACTIVATION_CLIPPED_RELU,
                                      CUDNN_NOT_PROPAGATE_NAN, 0.1);
}

TEST_F(ActivationTest, TestConstructorNoCoef) {
  Activation<float, float> activation(CUDNN_ACTIVATION_RELU,
                                      CUDNN_NOT_PROPAGATE_NAN);
}

TEST_F(ActivationTest, TestForward) {
  constexpr int n = 2;
  constexpr int c = 3;
  constexpr int h = 2;
  constexpr int w = 2;
  constexpr int n_elem = n * c * h * w;
  size_t n_bytes = sizeof(float) * n_elem;

  float x_host[n_elem] = {-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8,
                          -0.9, -1.0, -1.1, -1.2, 0.1,  0.2,  0.3,  0.4,
                          0.5,  0.6,  0.7,  0.8,  0.9,  1.0,  1.1,  1.2};
  float* x_dev;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, n_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(x_dev, x_host, n_bytes, cudaMemcpyHostToDevice));
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w, x_dev);

  float* y_dev;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_dev, n_bytes));
  Tensor<float> y(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w, y_dev);

  Activation<float, float> activation(CUDNN_ACTIVATION_RELU,
                                      CUDNN_NOT_PROPAGATE_NAN);
  activation.Forward(handle, 1, x, 0, &y);
  float y_host[n_elem] = {};
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(y_host, y_dev, n_bytes, cudaMemcpyDeviceToHost));
  float y_ref[n_elem] = {0,   0,   0,   0,   0,   0,   0,   0,
                         0,   0,   0,   0,   0.1, 0.2, 0.3, 0.4,
                         0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2};
  for (int i = 0; i < n_elem; ++i) {
    EXPECT_NEAR(y_ref[i], y_host[i], 1e-4) << "at index " << i;
  }
  CUDNNXX_CUDA_CHECK(cudaFree(y_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
}

TEST_F(ActivationTest, TestBackward) {
  constexpr int n = 2;
  constexpr int c = 3;
  constexpr int h = 2;
  constexpr int w = 2;
  constexpr int n_elem = n * c * h * w;
  size_t n_bytes = sizeof(float) * n_elem;

  float x_host[n_elem] = {-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8,
                          -0.9, -1.0, -1.1, -1.2, 0.1,  0.2,  0.3,  0.4,
                          0.5,  0.6,  0.7,  0.8,  0.9,  1.0,  1.1,  1.2};
  float* x_dev;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, n_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(x_dev, x_host, n_bytes, cudaMemcpyHostToDevice));
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w, x_dev);

  float y_host[n_elem] = {0,   0,   0,   0,   0,   0,   0,   0,
                          0,   0,   0,   0,   0.1, 0.2, 0.3, 0.4,
                          0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2};
  float* y_dev;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_dev, n_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(y_dev, y_host, n_bytes, cudaMemcpyHostToDevice));
  Tensor<float> y(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w, y_dev);

  float dy_host[n_elem] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  float* dy_dev;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dy_dev, n_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(dy_dev, dy_host, n_bytes, cudaMemcpyHostToDevice));
  Tensor<float> dy(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w, dy_dev);

  float* dx_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dx_dev, n_bytes));
  Tensor<float> dx(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w, dx_dev);

  Activation<float, float> activation(CUDNN_ACTIVATION_RELU,
                                      CUDNN_NOT_PROPAGATE_NAN);
  activation.Backward(handle, 1, y, dy, x, 0, &dx);
  float dx_host[n_elem] = {};
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(dx_host, dx_dev, n_bytes, cudaMemcpyDeviceToHost));
  float dx_ref[n_elem] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  for (int i = 0; i < n_elem; ++i) {
    EXPECT_NEAR(dx_ref[i], dx_host[i], 1e-4) << "at index " << i;
  }
  CUDNNXX_CUDA_CHECK(cudaFree(dx_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dy_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(y_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
}

}  // namespace cudnnxx
