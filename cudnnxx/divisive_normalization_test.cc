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
  size_t size_in_bytes = sizeof(float) * n_elem;

  float x_mem_host[n_elem] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                              0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                              1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4};
  float* x_mem_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_mem_dev, size_in_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(x_mem_dev, x_mem_host, size_in_bytes, cudaMemcpyHostToDevice));
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w, x_mem_dev);

  float means_mem_host[n_elem] = {};
  float* means_mem_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&means_mem_dev, size_in_bytes));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(means_mem_dev, means_mem_host, size_in_bytes,
                                cudaMemcpyHostToDevice));
  Tensor<float> means(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w,
                      means_mem_dev);

  float* temp_mem_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&temp_mem_dev, size_in_bytes));
  Tensor<float> temp(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w,
                     temp_mem_dev);

  float* temp2_mem_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&temp2_mem_dev, size_in_bytes));
  Tensor<float> temp2(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w,
                      temp2_mem_dev);

  float* y_mem_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_mem_dev, size_in_bytes));
  Tensor<float> y(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w, y_mem_dev);

  float alpha = 1;
  float beta = 0;
  dn.Forward(handle, alpha, x, means, temp, temp2, beta, &y);

  float y_mem_host[n_elem] = {};
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(y_mem_host, y_mem_dev, size_in_bytes, cudaMemcpyDeviceToHost));

  float* y_ref_mem_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_ref_mem_dev, size_in_bytes));
  Tensor<float> y_ref(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w,
                      y_ref_mem_dev);

  CUDNNXX_DNN_CHECK(cudnnDivisiveNormalizationForward(
      handle.raw_handle(), dn.desc(), CUDNN_DIVNORM_PRECOMPUTED_MEANS, &alpha,
      x.desc(), x.dev_mem(), means.dev_mem(), temp.dev_mem(), temp2.dev_mem(),
      &beta, y_ref.desc(), y_ref.dev_mem()));

  float y_ref_mem_host[n_elem] = {};
  CUDNNXX_CUDA_CHECK(cudaMemcpy(y_ref_mem_host, y_ref_mem_dev, size_in_bytes,
                                cudaMemcpyDeviceToHost));

  for (int i = 0; i < n_elem; ++i) {
    EXPECT_NEAR(y_ref_mem_host[i], y_mem_host[i], 1e-4) << "i: " << i;
  }

  CUDNNXX_CUDA_CHECK(cudaFree(y_ref_mem_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(y_mem_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(temp2_mem_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(temp_mem_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(means_mem_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(x_mem_dev));
}

}  // namespace cudnnxx
