#include "gtest/gtest.h"
#include "cuxx/dnn/op_tensor.h"

namespace cuxx {
namespace dnn {

class OpTensorTest : public ::testing::Test {
 protected:
  Handle handle;
};

TEST_F(OpTensorTest, TestConstructor) {
  OpTensor<float, float> op_tensor(CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT,
                     CUDNN_PROPAGATE_NAN);
}

TEST_F(OpTensorTest, TestDesc) {
  OpTensor<float, float> op_tensor(CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT,
                     CUDNN_PROPAGATE_NAN);
  cudnnOpTensorDescriptor_t desc = op_tensor.desc();
  CUXX_UNUSED_VAR(desc);
}

TEST_F(OpTensorTest, TestAdd) {
  constexpr int n = 2;
  constexpr int c = 3;
  constexpr int h = 2;
  constexpr int w = 2;
  constexpr int n_elem = n * c * h * w;
  size_t size = sizeof(float) * n_elem;

  float a_host[n_elem] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                          0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
                          1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
                          1.9, 2.0, 2.1, 2.2, 2.3, 2.4};
  float* a_dev;
  CUXX_CUDA_CHECK(cudaMalloc(&a_dev, size));
  CUXX_CUDA_CHECK(cudaMemcpy(a_dev, a_host, size, cudaMemcpyHostToDevice));
  Tensor<float> a_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w,
                         a_dev);

  float b_host[n_elem] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                          0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
                          1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
                          1.9, 2.0, 2.1, 2.2, 2.3, 2.4};
  float* b_dev;
  CUXX_CUDA_CHECK(cudaMalloc(&b_dev, size));
  CUXX_CUDA_CHECK(cudaMemcpy(b_dev, b_host, size, cudaMemcpyHostToDevice));
  Tensor<float> b_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w,
                         b_dev);

  float c_host[n_elem] = {};
  float* c_dev;
  CUXX_CUDA_CHECK(cudaMalloc(&c_dev, size));
  CUXX_CUDA_CHECK(cudaMemcpy(c_dev, c_host, size, cudaMemcpyHostToDevice));
  Tensor<float> c_tensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w,
                         c_dev);

  OpTensor<float, float> op_tensor(CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT,
                                   CUDNN_PROPAGATE_NAN);
  op_tensor.Compute(handle, 1, a_tensor, 1, b_tensor, 0, &c_tensor);
  CUXX_CUDA_CHECK(cudaMemcpy(c_host, c_dev, size, cudaMemcpyDeviceToHost));

  float c_expected[n_elem] = {0.2, 0.4, 0.6, 0.8, 1.0, 1.2,
                              1.4, 1.6, 1.8, 2.0, 2.2, 2.4,
                              2.6, 2.8, 3.0, 3.2, 3.4, 3.6,
                              3.8, 4.0, 4.2, 4.4, 4.6, 4.8};
  for (int i = 0; i < n_elem; ++i) {
    EXPECT_NEAR(c_expected[i], c_host[i], 1e-4) << "i: " << i;
  }
  CUXX_CUDA_CHECK(cudaFree(c_dev));
  CUXX_CUDA_CHECK(cudaFree(b_dev));
  CUXX_CUDA_CHECK(cudaFree(a_dev));
}

}  // namespace dnn
}  // namespace cuxx
