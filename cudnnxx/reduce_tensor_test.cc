#include "cudnnxx/reduce_tensor.h"

#include "gtest/gtest.h"

namespace cudnnxx {

class ReduceTensorTest : public ::testing::Test {
 protected:
  Handle handle;
};

TEST_F(ReduceTensorTest, TestConstructor) {
  ReduceTensor<float, float> reduce_tensor(
      CUDNN_REDUCE_TENSOR_MAX, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN,
      CUDNN_REDUCE_TENSOR_FLATTENED_INDICES, CUDNN_32BIT_INDICES);
}

TEST_F(ReduceTensorTest, TestCompute) {
  constexpr int n = 2;
  constexpr int c = 3;
  constexpr int h = 2;
  constexpr int w = 2;
  constexpr int n_elem = n * c * h * w;
  size_t size = sizeof(float) * n_elem;

  float a_host[n_elem] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                          0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                          1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4};
  float* a_dev;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&a_dev, size));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(a_dev, a_host, size, cudaMemcpyHostToDevice));
  Tensor<float> a_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w,
                         a_dev);

  float* c_dev;
  size_t c_dev_size = sizeof(float);
  CUDNNXX_CUDA_CHECK(cudaMalloc(&c_dev, c_dev_size));
  Tensor<float> c_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, 1, 1,
                         c_dev);

  ReduceTensor<float, float> reduce_tensor(
      CUDNN_REDUCE_TENSOR_MAX, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN,
      CUDNN_REDUCE_TENSOR_FLATTENED_INDICES, CUDNN_32BIT_INDICES);

  size_t indices_size_in_bytes = 0;
  CUDNNXX_DNN_CHECK(cudnnGetReductionIndicesSize(
      handle.raw_handle(), reduce_tensor.desc(), a_tensor.desc(),
      c_tensor.desc(), &indices_size_in_bytes));
  void* indices_dev;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&indices_dev, indices_size_in_bytes));

  size_t workspace_size_in_bytes = 0;
  CUDNNXX_DNN_CHECK(cudnnGetReductionWorkspaceSize(
      handle.raw_handle(), reduce_tensor.desc(), a_tensor.desc(),
      c_tensor.desc(), &workspace_size_in_bytes));
  void* workspace;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&workspace, workspace_size_in_bytes));

  reduce_tensor.Compute(handle, indices_dev, indices_size_in_bytes, workspace,
                        workspace_size_in_bytes, 1, a_tensor, 0, &c_tensor);
  uint8_t indices_host = 0;
  CUDNNXX_CUDA_CHECK(cudaMemcpy(&indices_host, indices_dev,
                                indices_size_in_bytes, cudaMemcpyDeviceToHost));
  float c_host = 0;
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(&c_host, c_dev, c_dev_size, cudaMemcpyDeviceToHost));

  EXPECT_EQ(23, indices_host);
  EXPECT_NEAR(2.4, c_host, 1e-4);
  CUDNNXX_CUDA_CHECK(cudaFree(workspace));
  CUDNNXX_CUDA_CHECK(cudaFree(indices_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(c_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(a_dev));
}

}  // namespace cudnnxx
