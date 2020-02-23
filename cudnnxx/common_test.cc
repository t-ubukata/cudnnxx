#include "gtest/gtest.h"
#include "cudnnxx/common.h"

namespace cudnnxx {

class HandleTest : public ::testing::Test {};

TEST_F(HandleTest, TestConstructor) {
  Handle handle;
}

TEST_F(HandleTest, TestRawHandle) {
  Handle handle;
  cudnnHandle_t raw_handle = handle.raw_handle();
  CUDNNXX_UNUSED_VAR(raw_handle);
}

class TensorTest : public ::testing::Test {};

TEST_F(TensorTest, TestConstructor4d) {
  constexpr int n = 2;
  constexpr int c = 3;
  constexpr int h = 2;
  constexpr int w = 2;
  constexpr int n_elem = n * c * h * w;
  float mem_host[n_elem] = {};
  float* mem_dev = nullptr;
  size_t size = sizeof(float) * n_elem;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&mem_dev, size));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(mem_dev, mem_host, size, cudaMemcpyHostToDevice));
  Tensor<float> t(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w, mem_dev);
  CUDNNXX_CUDA_CHECK(cudaFree(mem_dev));
}

TEST_F(TensorTest, TestConstructor4dEx) {
  constexpr int n = 2;
  constexpr int c = 3;
  constexpr int h = 2;
  constexpr int w = 2;
  constexpr int n_elem = n * c * h * w;
  float* mem_host[n_elem] = {};
  float* mem_dev = nullptr;
  size_t size = sizeof(float) * n_elem;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&mem_dev, size));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(mem_dev, mem_host, size, cudaMemcpyHostToDevice));
  // NHWC
  int n_stride = 12;
  int c_stride = 1;
  int h_stride = 6;
  int w_stride = 3;
  Tensor<float> t(CUDNN_DATA_FLOAT, n, c, h, w,
                  n_stride, c_stride, h_stride, w_stride, mem_dev);
  CUDNNXX_CUDA_CHECK(cudaFree(mem_dev));
}

TEST_F(TensorTest, TestConstructorNd) {
  constexpr int n_dims = 3;
  constexpr int n = 4;
  constexpr int h = 2;
  constexpr int w = 2;
  constexpr int n_elem = n * h * w;
  float* mem_host[n_elem] = {};
  float* mem_dev = nullptr;
  size_t size = sizeof(float) * n_elem;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&mem_dev, size));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(mem_dev, mem_host, size, cudaMemcpyHostToDevice));
  // HWN
  int dims[n_dims] = {n, h, w};
  int strides[n_dims] = {1, 8, 4};
  Tensor<float> t(CUDNN_DATA_FLOAT, n_dims, dims, strides, mem_dev);
  CUDNNXX_CUDA_CHECK(cudaFree(mem_dev));
}

TEST_F(TensorTest, TestConstructorNdEx) {
  constexpr int n_dims = 4;
  constexpr int n = 2;
  constexpr int c = 3;
  constexpr int h = 2;
  constexpr int w = 2;
  constexpr int n_elem = n * c * h * w;
  float* mem_host[n_elem] = {};
  float* mem_dev = nullptr;
  size_t size = sizeof(float) * n_elem;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&mem_dev, size));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(mem_dev, mem_host, size, cudaMemcpyHostToDevice));
  int dims[n_dims] = {n, c, h, w};
  Tensor<float> t(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_dims, dims, mem_dev);
  CUDNNXX_CUDA_CHECK(cudaFree(mem_dev));
}

class FilterTest : public ::testing::Test {};

TEST_F(FilterTest, TestConstructor4d) {
  constexpr int k = 1;  // the number of output feature maps.
  constexpr int c = 3;  // the number of input feature maps.
  constexpr int h = 5;  // The height of each filter.
  constexpr int w = 5;  // The width of each filter.
  constexpr int n_elem = k * c * h * w;
  float* mem_host[n_elem] = {};
  float* mem_dev = nullptr;
  size_t size = sizeof(float) * n_elem;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&mem_dev, size));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(mem_dev, mem_host, size, cudaMemcpyHostToDevice));
  Filter<float> f(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, h, w, mem_dev);
  CUDNNXX_CUDA_CHECK(cudaFree(mem_dev));
}

TEST_F(FilterTest, TestConstructorNd) {
  constexpr int k = 1;  // the number of output feature maps.
  constexpr int c = 3;  // the number of input feature maps.
  constexpr int r = 5;  // The number of rows per filter.
  constexpr int s = 5;  // The number of columns per filter.
  constexpr int n_elem = k * c * r * s;
  float* mem_host[n_elem] = {};
  float* mem_dev = nullptr;
  size_t size = sizeof(float) * n_elem;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&mem_dev, size));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(mem_dev, mem_host, size, cudaMemcpyHostToDevice));
  constexpr int n_dims = 4;
  int dims[n_dims] = {k, c, r, s};
  Filter<float> f(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n_dims, dims, mem_dev);
  CUDNNXX_CUDA_CHECK(cudaFree(mem_dev));
}

}  // namespace cudnnxx