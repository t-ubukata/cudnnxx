#include "gtest/gtest.h"
#include "cuxx/dnn/common.h"

namespace cuxx {
namespace dnn {

class HandleTest : public ::testing::Test {};

TEST_F(HandleTest, TestConstructor) {
  Handle handle;
}

TEST_F(HandleTest, TestRawHandle) {
  Handle handle;
  cudnnHandle_t raw_handle = handle.raw_handle();
  CUXX_UNUSED_VAR(raw_handle);
}

class TensorTest : public ::testing::Test {};

TEST_F(TensorTest, TestConstructor4d) {
  constexpr int n = 2;
  constexpr int c = 3;
  constexpr int h = 2;
  constexpr int w = 2;
  constexpr int num_elem = n * c * h * w;
  float* mem_host[num_elem] = {};
  float* mem_dev = nullptr;
  size_t size = sizeof(float) * num_elem;
  cudaMalloc(&mem_dev, size);
  cudaMemcpy(mem_host, mem_dev, size, cudaMemcpyHostToDevice);
  Tensor<float> t(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w, mem_dev);
  cudaFree(mem_dev);
}

TEST_F(TensorTest, TestConstructor4dEx) {
  constexpr int n = 2;
  constexpr int c = 3;
  constexpr int h = 2;
  constexpr int w = 2;
  constexpr int num_elem = n * c * h * w;
  float* mem_host[num_elem] = {};
  float* mem_dev = nullptr;
  size_t size = sizeof(float) * num_elem;
  cudaMalloc(&mem_dev, size);
  cudaMemcpy(mem_host, mem_dev, size, cudaMemcpyHostToDevice);
  // NHWC
  int n_stride = 12;
  int c_stride = 1;
  int h_stride = 6;
  int w_stride = 3;
  Tensor<float> t(CUDNN_DATA_FLOAT, n, c, h, w,
                  n_stride, c_stride, h_stride, w_stride, mem_dev);
  cudaFree(mem_dev);
}

TEST_F(TensorTest, TestConstructorNd) {
  constexpr int nbDims = 3;
  constexpr int n = 4;
  constexpr int h = 2;
  constexpr int w = 2;
  constexpr int num_elem = n * h * w;
  float* mem_host[num_elem] = {};
  float* mem_dev = nullptr;
  size_t size = sizeof(float) * num_elem;
  cudaMalloc(&mem_dev, size);
  cudaMemcpy(mem_host, mem_dev, sizeof(float) * num_elem,
             cudaMemcpyHostToDevice);
  // HWN
  int dimA[nbDims] = {n, h, w};
  int strideA[nbDims] = {1, 8, 4};
  Tensor<float> t(CUDNN_DATA_FLOAT, nbDims, dimA, strideA, mem_dev);
  cudaFree(mem_dev);
}

TEST_F(TensorTest, TestConstructorNdEx) {
  constexpr int nbDims = 4;
  constexpr int n = 2;
  constexpr int c = 3;
  constexpr int h = 2;
  constexpr int w = 2;
  constexpr int num_elem = n * c * h * w;
  float* mem_host[num_elem] = {};
  float* mem_dev = nullptr;
  size_t size = sizeof(float) * num_elem;
  cudaMalloc(&mem_dev, size);
  cudaMemcpy(mem_host, mem_dev, sizeof(float) * num_elem,
             cudaMemcpyHostToDevice);
  int dimA[nbDims] = {n, c, h, w};
  Tensor<float> t(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, nbDims, dimA, mem_dev);
  cudaFree(mem_dev);
}

}  // namespace dnn
}  // namespace cuxx
