#include "cudnnxx/divisive_normalization.h"

#include "gtest/gtest.h"

namespace cudnnxx {

class DivisiveNormalizationTest : public ::testing::Test {
 protected:
  Handle handle;
};

TEST_F(DivisiveNormalizationTest, TestConstructor) {
  DivisiveNormalization<float, float> dn(5, 1e-4, 0.75, 2);
}

TEST_F(DivisiveNormalizationTest, TestForward) {
  DivisiveNormalization<float, float> dn(5, 1e-4, 0.75, 2);

  constexpr int n = 2;
  constexpr int c = 3;
  constexpr int h = 2;
  constexpr int w = 2;
  constexpr int n_elem = n * c * h * w;
  size_t n_bytes = sizeof(float) * n_elem;

  float x_host[n_elem] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                              0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                              1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4};
  float* x_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, n_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(x_dev, x_host, n_bytes, cudaMemcpyHostToDevice));
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w, x_dev);

  float means_host[n_elem] = {};
  float* means_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&means_dev, n_bytes));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(means_dev, means_host, n_bytes,
                                cudaMemcpyHostToDevice));
  Tensor<float> means(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w,
                      means_dev);

  float* temp_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&temp_dev, n_bytes));
  Tensor<float> temp(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w,
                     temp_dev);

  float* temp2_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&temp2_dev, n_bytes));
  Tensor<float> temp2(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w,
                      temp2_dev);

  float* y_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_dev, n_bytes));
  Tensor<float> y(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w, y_dev);

  float alpha = 1;
  float beta = 0;
  dn.Forward(handle, alpha, x, means, temp, temp2, beta, &y);

  float y_host[n_elem] = {};
  CUDNNXX_CUDA_CHECK(cudaMemcpy(y_host, y_dev, n_bytes, cudaMemcpyDeviceToHost));

  float* y_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_ref_dev, n_bytes));
  Tensor<float> y_ref(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w,
                      y_ref_dev);

  CUDNNXX_DNN_CHECK(cudnnDivisiveNormalizationForward(
      handle.raw_handle(), dn.desc(), CUDNN_DIVNORM_PRECOMPUTED_MEANS, &alpha,
      x.desc(), x.dev_mem(), means.dev_mem(), temp.dev_mem(), temp2.dev_mem(),
      &beta, y_ref.desc(), y_ref.dev_mem()));

  float y_ref_host[n_elem] = {};
  CUDNNXX_CUDA_CHECK(cudaMemcpy(y_ref_host, y_ref_dev, n_bytes,
                                cudaMemcpyDeviceToHost));

  for (int i = 0; i < n_elem; ++i) {
    EXPECT_EQ(y_ref_host[i], y_host[i]) << "at index " << i;
  }

  CUDNNXX_CUDA_CHECK(cudaFree(y_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(y_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(temp2_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(temp_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(means_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
}

TEST_F(DivisiveNormalizationTest, TestBackward) {
  DivisiveNormalization<float, float> dn(5, 1e-4, 0.75, 2);

  constexpr int n = 2;
  constexpr int c = 3;
  constexpr int h = 2;
  constexpr int w = 2;
  constexpr int n_elem = n * c * h * w;
  size_t n_bytes = sizeof(float) * n_elem;

  float x_host[n_elem] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                              0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                              1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4};
  float* x_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, n_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(x_dev, x_host, n_bytes, cudaMemcpyHostToDevice));
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w, x_dev);

  float means_host[n_elem] = {};
  float* means_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&means_dev, n_bytes));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(means_dev, means_host, n_bytes,
                                cudaMemcpyHostToDevice));
  Tensor<float> means(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w,
                      means_dev);

  float* temp_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&temp_dev, n_bytes));
  Tensor<float> temp(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w, temp_dev);

  float* temp2_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&temp2_dev, n_bytes));
  Tensor<float> temp2(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w,
                      temp2_dev);

  float dy_host[n_elem] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                               0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                               1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4};
  float* dy_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dy_dev, n_bytes));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(dy_dev, dy_host, n_bytes, cudaMemcpyHostToDevice));
  Tensor<float> dy(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w, dy_dev);

  float* dx_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dx_dev, n_bytes));
  Tensor<float> dx(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w, dx_dev);

  float* d_means_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&d_means_dev, n_bytes));
  Tensor<float> d_means(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w,
                        d_means_dev);

  float alpha = 1;
  float beta = 0;
  dn.Backward(handle, alpha, x, means, dy, temp, temp2, beta, &dx, &d_means);

  float dx_host[n_elem] = {};
  CUDNNXX_CUDA_CHECK(cudaMemcpy(dx_host, dx_dev, n_bytes, cudaMemcpyDeviceToHost));

  float d_means_host[n_elem] = {};
  CUDNNXX_CUDA_CHECK(cudaMemcpy(d_means_host, d_means_dev, n_bytes, cudaMemcpyDeviceToHost));

  float* dx_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dx_ref_dev, n_bytes));
  Tensor<float> dx_ref(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w, dx_ref_dev);

  float* d_means_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&d_means_ref_dev, n_bytes));
  Tensor<float> d_means_ref(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w,
                            d_means_ref_dev);

  CUDNNXX_DNN_CHECK(cudnnDivisiveNormalizationBackward(
      handle.raw_handle(), dn.desc(), CUDNN_DIVNORM_PRECOMPUTED_MEANS, &alpha,
      x.desc(), x.dev_mem(), means.dev_mem(), dy.dev_mem(), temp.dev_mem(),
      temp2.dev_mem(), &beta, dx_ref.desc(), dx_ref.dev_mem(),
      d_means_ref.dev_mem()));

  float dx_ref_host[n_elem] = {};
  CUDNNXX_CUDA_CHECK(cudaMemcpy(dx_ref_host, dx_ref_dev, n_bytes, cudaMemcpyDeviceToHost));

  float d_means_ref_host[n_elem] = {};
  CUDNNXX_CUDA_CHECK(cudaMemcpy(d_means_ref_host, d_means_ref_dev,
                                n_bytes, cudaMemcpyDeviceToHost));

  for (int i = 0; i < n_elem; ++i) {
    EXPECT_EQ(dx_ref_host[i], dx_host[i]) << "at index " << i;
    EXPECT_EQ(d_means_ref_host[i], d_means_host[i]) << "at index " << i;
  }

  CUDNNXX_CUDA_CHECK(cudaFree(d_means_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dx_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(d_means_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dx_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dy_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(temp2_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(temp_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(means_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
}

}  // namespace cudnnxx
