#include "cudnnxx/rnn.h"

#include <algorithm>
#include <vector>

#include "cudnnxx/dropout.h"
#include "gtest/gtest.h"

namespace cudnnxx {

class RNNTest : public ::testing::Test {
 protected:
  const Handle handle;
  const float dropout_p = 0.5;
  const unsigned long long seed = 20200627;
  const cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
};

TEST_F(RNNTest, TestConstructor) {
  Dropout<float> dropout(handle, dropout_p, seed);
  RNN<float> rnn(handle, 1, 1, dropout, CUDNN_LINEAR_INPUT,
                 CUDNN_UNIDIRECTIONAL, CUDNN_RNN_RELU, CUDNN_RNN_ALGO_STANDARD,
                 CUDNN_DATA_FLOAT);
}

TEST_F(RNNTest, TestGetParamSize) {
  Dropout<float> dropout(handle, dropout_p, seed);
  RNN<float> rnn(handle, 1, 1, dropout, CUDNN_LINEAR_INPUT,
                 CUDNN_UNIDIRECTIONAL, CUDNN_RNN_RELU, CUDNN_RNN_ALGO_STANDARD,
                 dtype);

  constexpr int n_dims = 3;
  constexpr int batch_n_elem = 2;
  constexpr int input_n_elem = 3;
  int dims[n_dims] = {batch_n_elem, input_n_elem, 1};
  int strides[n_dims] = {dims[2] * dims[1], dims[2], 1};
  constexpr int seq_length = 3;
  constexpr int n_elem = seq_length * input_n_elem * batch_n_elem;

  float x_host[n_elem] = {};
  float* x_dev = nullptr;
  size_t x_size_in_bytes = sizeof(float) * n_elem;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, x_size_in_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(x_dev, x_host, x_size_in_bytes, cudaMemcpyHostToDevice));
  TensorArray<float> x_tensors(dtype, n_dims, dims, strides, x_dev, seq_length);

  auto size_in_bytes = rnn.GetParamsSize(handle, x_tensors, dtype);

  size_t size_in_bytes_ref = 0;
  CUDNNXX_DNN_CHECK(cudnnGetRNNParamsSize(handle.raw_handle(), rnn.desc(),
                                          x_tensors.descs()[0],
                                          &size_in_bytes_ref, dtype));

  EXPECT_EQ(size_in_bytes_ref, size_in_bytes);
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
}

TEST_F(RNNTest, TestGetTrainingReserveSize) {
  Dropout<float> dropout(handle, dropout_p, seed);
  RNN<float> rnn(handle, 1, 1, dropout, CUDNN_LINEAR_INPUT,
                 CUDNN_UNIDIRECTIONAL, CUDNN_RNN_RELU, CUDNN_RNN_ALGO_STANDARD,
                 dtype);

  constexpr int n_dims = 3;
  constexpr int batch_n_elem = 2;
  constexpr int input_n_elem = 3;
  int dims[n_dims] = {batch_n_elem, input_n_elem, 1};
  int strides[n_dims] = {dims[2] * dims[1], dims[2], 1};
  constexpr int seq_length = 3;
  constexpr int n_elem = seq_length * input_n_elem * batch_n_elem;

  float x_host[n_elem] = {};
  float* x_dev = nullptr;
  size_t x_size_in_bytes = sizeof(float) * n_elem;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, x_size_in_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(x_dev, x_host, x_size_in_bytes, cudaMemcpyHostToDevice));
  TensorArray<float> x_tensors(dtype, n_dims, dims, strides, x_dev, seq_length);

  auto reserve_size_in_bytes =
      rnn.GetTrainingReserveSize(handle, seq_length, x_tensors);

  size_t reserve_size_in_bytes_ref = 0;
  CUDNNXX_DNN_CHECK(cudnnGetRNNTrainingReserveSize(
      handle.raw_handle(), rnn.desc(), seq_length, x_tensors.descs(),
      &reserve_size_in_bytes_ref));

  EXPECT_EQ(reserve_size_in_bytes_ref, reserve_size_in_bytes);
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
}

TEST_F(RNNTest, TestGetRNNWorkspaceSize) {
  Dropout<float> dropout(handle, dropout_p, seed);
  RNN<float> rnn(handle, 1, 1, dropout, CUDNN_LINEAR_INPUT,
                 CUDNN_UNIDIRECTIONAL, CUDNN_RNN_RELU, CUDNN_RNN_ALGO_STANDARD,
                 dtype);

  constexpr int n_dims = 3;
  constexpr int batch_n_elem = 2;
  constexpr int input_n_elem = 3;
  int dims[n_dims] = {batch_n_elem, input_n_elem, 1};
  int strides[n_dims] = {dims[2] * dims[1], dims[2], 1};
  constexpr int seq_length = 3;
  constexpr int n_elem = seq_length * input_n_elem * batch_n_elem;

  float x_host[n_elem] = {};
  float* x_dev = nullptr;
  size_t x_size_in_bytes = sizeof(float) * n_elem;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, x_size_in_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(x_dev, x_host, x_size_in_bytes, cudaMemcpyHostToDevice));
  TensorArray<float> x_tensors(dtype, n_dims, dims, strides, x_dev, seq_length);

  auto workspace_size_in_bytes =
      rnn.GetWorkspaceSize(handle, seq_length, x_tensors);

  size_t workspace_size_in_bytes_ref = 0;
  CUDNNXX_DNN_CHECK(cudnnGetRNNWorkspaceSize(handle.raw_handle(), rnn.desc(),
                                             seq_length, x_tensors.descs(),
                                             &workspace_size_in_bytes_ref));

  EXPECT_EQ(workspace_size_in_bytes_ref, workspace_size_in_bytes);
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
}

TEST_F(RNNTest, TestForwardTraining) {
  Dropout<float> dropout(handle, dropout_p, seed);
  int input_n_elem = 3;
  int hidden_n_elem = input_n_elem;
  int n_layers = 1;
  RNN<float> rnn(handle, hidden_n_elem, n_layers, dropout, CUDNN_LINEAR_INPUT,
                 CUDNN_UNIDIRECTIONAL, CUDNN_RNN_RELU, CUDNN_RNN_ALGO_STANDARD,
                 dtype);

  int n_dims = 3;
  int batch_n_elem = 2;
  std::vector<int> dims_1 = {batch_n_elem, input_n_elem, 1};
  std::vector<int> strides_1 = {dims_1[2] * dims_1[1], dims_1[2], 1};
  int seq_length = 3;
  int n_elem_1 = dims_1[0] * dims_1[1] * dims_1[2] * seq_length;
  size_t size_in_bytes_1 = sizeof(float) * n_elem_1;

  // x
  std::vector<float> x_host = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                               1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8};
  float* x_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, size_in_bytes_1));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host.data(), size_in_bytes_1,
                                cudaMemcpyHostToDevice));
  TensorArray<float> x_tensors(dtype, n_dims, dims_1.data(), strides_1.data(),
                               x_dev, seq_length);

  // y
  float* y_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_dev, size_in_bytes_1));
  TensorArray<float> y_tensors(dtype, n_dims, dims_1.data(), strides_1.data(),
                               y_dev, seq_length);

  std::vector<int> dims_2 = {n_layers, batch_n_elem, hidden_n_elem};
  std::vector<int> strides_2 = {dims_2[2] * dims_2[1], dims_2[2], 1};
  int n_elem_2 = dims_2[0] * dims_2[1] * dims_2[2];
  size_t size_in_bytes_2 = sizeof(float) * n_elem_2;

  // hx
  std::vector<float> hx_host = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  float* hx_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&hx_dev, size_in_bytes_2));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(hx_dev, hx_host.data(), size_in_bytes_2,
                                cudaMemcpyHostToDevice));
  Tensor<float> hx_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                          hx_dev);

  // cx
  std::vector<float> cx_host = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  float* cx_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&cx_dev, size_in_bytes_2));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(cx_dev, cx_host.data(), size_in_bytes_2,
                                cudaMemcpyHostToDevice));
  Tensor<float> cx_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                          cx_dev);

  // hy
  float* hy_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&hy_dev, size_in_bytes_2));
  Tensor<float> hy_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                          hy_dev);

  // cy
  float* cy_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&cy_dev, size_in_bytes_2));
  Tensor<float> cy_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                          cy_dev);

  // w
  auto w_size_in_bytes = rnn.GetParamsSize(handle, x_tensors, dtype);
  std::vector<int> w_dims = {static_cast<int>(w_size_in_bytes / sizeof(float)),
                             1, 1};
  std::vector<float> w_host(w_dims[0] * w_dims[1] * w_dims[2]);
  w_host = {1};
  float* w_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&w_dev, w_size_in_bytes));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(w_dev, w_host.data(), w_size_in_bytes,
                                cudaMemcpyHostToDevice));
  Filter<float> w_filter(dtype, CUDNN_TENSOR_NCHW, n_dims, w_dims.data(),
                         w_dev);

  // workspace
  auto workspace_size_in_bytes =
      rnn.GetWorkspaceSize(handle, seq_length, x_tensors);
  void* workspace = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&workspace, workspace_size_in_bytes));

  // reserve_space
  auto reserve_size_in_bytes =
      rnn.GetTrainingReserveSize(handle, seq_length, x_tensors);
  void* reserve_space = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&reserve_space, reserve_size_in_bytes));

  // Compute.
  rnn.ForwardTraining(handle, seq_length, x_tensors, hx_tensor, cx_tensor,
                      w_filter, &y_tensors, &hy_tensor, &cy_tensor, workspace,
                      workspace_size_in_bytes, reserve_space,
                      reserve_size_in_bytes);

  // y_host
  std::vector<float> y_host(n_elem_1);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(y_host.data(), y_dev, size_in_bytes_1,
                                cudaMemcpyDeviceToHost));

  // hy_host
  std::vector<float> hy_host(n_elem_2);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(hy_host.data(), hy_dev, size_in_bytes_2,
                                cudaMemcpyDeviceToHost));

  // cy_host
  std::vector<float> cy_host(n_elem_2);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(cy_host.data(), cy_dev, size_in_bytes_2,
                                cudaMemcpyDeviceToHost));

  // y_ref
  float* y_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_ref_dev, size_in_bytes_1));
  TensorArray<float> y_ref_tensors(dtype, n_dims, dims_1.data(),
                                   strides_1.data(), y_ref_dev, seq_length);

  // hy_ref
  float* hy_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&hy_ref_dev, size_in_bytes_2));
  Tensor<float> hy_ref_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                              hy_ref_dev);

  // cy_ref
  float* cy_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&cy_ref_dev, size_in_bytes_2));
  Tensor<float> cy_ref_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                              cy_ref_dev);

  // Compute reference.
  CUDNNXX_DNN_CHECK(cudnnRNNForwardTraining(
      handle.raw_handle(), rnn.desc(), seq_length, x_tensors.descs(),
      x_tensors.dev_mem(), hx_tensor.desc(), hx_tensor.dev_mem(),
      cx_tensor.desc(), cx_tensor.dev_mem(), w_filter.desc(),
      w_filter.dev_mem(), y_ref_tensors.descs(), y_ref_tensors.dev_mem(),
      hy_ref_tensor.desc(), hy_ref_tensor.dev_mem(), cy_ref_tensor.desc(),
      cy_ref_tensor.dev_mem(), workspace, workspace_size_in_bytes,
      reserve_space, reserve_size_in_bytes));

  // y_ref_host
  std::vector<float> y_ref_host(n_elem_1);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(y_ref_host.data(), y_ref_dev, size_in_bytes_1,
                                cudaMemcpyDeviceToHost));

  // hy_ref_host
  std::vector<float> hy_ref_host(n_elem_2);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(hy_ref_host.data(), hy_ref_dev, size_in_bytes_2,
                                cudaMemcpyDeviceToHost));

  // cy_ref_host
  std::vector<float> cy_ref_host(n_elem_2);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(cy_ref_host.data(), cy_ref_dev, size_in_bytes_2,
                                cudaMemcpyDeviceToHost));

  // Check y, hy, cy.
  EXPECT_EQ(y_ref_host, y_host);
  EXPECT_EQ(hy_ref_host, hy_host);
  EXPECT_EQ(cy_ref_host, cy_host);

  CUDNNXX_CUDA_CHECK(cudaFree(cy_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(hy_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(y_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(reserve_space));
  CUDNNXX_CUDA_CHECK(cudaFree(workspace));
  CUDNNXX_CUDA_CHECK(cudaFree(w_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(cy_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(hy_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(cx_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(hx_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(y_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
}

TEST_F(RNNTest, TestForwardInference) {
  Dropout<float> dropout(handle, dropout_p, seed);
  int input_n_elem = 3;
  int hidden_n_elem = input_n_elem;
  int n_layers = 1;
  RNN<float> rnn(handle, hidden_n_elem, n_layers, dropout, CUDNN_LINEAR_INPUT,
                 CUDNN_UNIDIRECTIONAL, CUDNN_RNN_RELU, CUDNN_RNN_ALGO_STANDARD,
                 dtype);

  int n_dims = 3;
  int batch_n_elem = 2;
  std::vector<int> dims_1 = {batch_n_elem, input_n_elem, 1};
  std::vector<int> strides_1 = {dims_1[2] * dims_1[1], dims_1[2], 1};
  int seq_length = 3;
  int n_elem_1 = dims_1[0] * dims_1[1] * dims_1[2] * seq_length;
  size_t size_in_bytes_1 = sizeof(float) * n_elem_1;

  // x
  std::vector<float> x_host = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                               1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8};
  float* x_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, size_in_bytes_1));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host.data(), size_in_bytes_1,
                                cudaMemcpyHostToDevice));
  TensorArray<float> x_tensors(dtype, n_dims, dims_1.data(), strides_1.data(),
                               x_dev, seq_length);

  // y
  float* y_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_dev, size_in_bytes_1));
  TensorArray<float> y_tensors(dtype, n_dims, dims_1.data(), strides_1.data(),
                               y_dev, seq_length);

  std::vector<int> dims_2 = {n_layers, batch_n_elem, hidden_n_elem};
  std::vector<int> strides_2 = {dims_2[2] * dims_2[1], dims_2[2], 1};
  int n_elem_2 = dims_2[0] * dims_2[1] * dims_2[2];
  size_t size_in_bytes_2 = sizeof(float) * n_elem_2;

  // hx
  std::vector<float> hx_host = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  float* hx_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&hx_dev, size_in_bytes_2));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(hx_dev, hx_host.data(), size_in_bytes_2,
                                cudaMemcpyHostToDevice));
  Tensor<float> hx_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                          hx_dev);

  // cx
  std::vector<float> cx_host = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  float* cx_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&cx_dev, size_in_bytes_2));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(cx_dev, cx_host.data(), size_in_bytes_2,
                                cudaMemcpyHostToDevice));
  Tensor<float> cx_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                          cx_dev);

  // hy
  float* hy_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&hy_dev, size_in_bytes_2));
  Tensor<float> hy_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                          hy_dev);

  // cy
  float* cy_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&cy_dev, size_in_bytes_2));
  Tensor<float> cy_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                          cy_dev);

  // w
  auto w_size_in_bytes = rnn.GetParamsSize(handle, x_tensors, dtype);
  std::vector<int> w_dims = {static_cast<int>(w_size_in_bytes / sizeof(float)),
                             1, 1};
  std::vector<float> w_host(w_dims[0] * w_dims[1] * w_dims[2]);
  w_host = {1};
  float* w_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&w_dev, w_size_in_bytes));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(w_dev, w_host.data(), w_size_in_bytes,
                                cudaMemcpyHostToDevice));
  Filter<float> w_filter(dtype, CUDNN_TENSOR_NCHW, n_dims, w_dims.data(),
                         w_dev);

  // workspace
  auto workspace_size_in_bytes =
      rnn.GetWorkspaceSize(handle, seq_length, x_tensors);
  void* workspace = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&workspace, workspace_size_in_bytes));

  // Compute.
  rnn.ForwardInference(handle, seq_length, x_tensors, hx_tensor, cx_tensor,
                       w_filter, &y_tensors, &hy_tensor, &cy_tensor, workspace,
                       workspace_size_in_bytes);

  // y_host
  std::vector<float> y_host(n_elem_1);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(y_host.data(), y_dev, size_in_bytes_1,
                                cudaMemcpyDeviceToHost));

  // hy_host
  std::vector<float> hy_host(n_elem_2);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(hy_host.data(), hy_dev, size_in_bytes_2,
                                cudaMemcpyDeviceToHost));

  // cy_host
  std::vector<float> cy_host(n_elem_2);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(cy_host.data(), cy_dev, size_in_bytes_2,
                                cudaMemcpyDeviceToHost));

  // y_ref
  float* y_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_ref_dev, size_in_bytes_1));
  TensorArray<float> y_ref_tensors(dtype, n_dims, dims_1.data(),
                                   strides_1.data(), y_ref_dev, seq_length);

  // hy_ref
  float* hy_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&hy_ref_dev, size_in_bytes_2));
  Tensor<float> hy_ref_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                              hy_ref_dev);

  // cy_ref
  float* cy_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&cy_ref_dev, size_in_bytes_2));
  Tensor<float> cy_ref_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                              cy_ref_dev);

  // Compute reference.
  CUDNNXX_DNN_CHECK(cudnnRNNForwardInference(
      handle.raw_handle(), rnn.desc(), seq_length, x_tensors.descs(),
      x_tensors.dev_mem(), hx_tensor.desc(), hx_tensor.dev_mem(),
      cx_tensor.desc(), cx_tensor.dev_mem(), w_filter.desc(),
      w_filter.dev_mem(), y_ref_tensors.descs(), y_ref_tensors.dev_mem(),
      hy_ref_tensor.desc(), hy_ref_tensor.dev_mem(), cy_ref_tensor.desc(),
      cy_ref_tensor.dev_mem(), workspace, workspace_size_in_bytes));

  // y_ref_host
  std::vector<float> y_ref_host(n_elem_1);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(y_ref_host.data(), y_ref_dev, size_in_bytes_1,
                                cudaMemcpyDeviceToHost));

  // hy_ref_host
  std::vector<float> hy_ref_host(n_elem_2);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(hy_ref_host.data(), hy_ref_dev, size_in_bytes_2,
                                cudaMemcpyDeviceToHost));

  // cy_ref_host
  std::vector<float> cy_ref_host(n_elem_2);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(cy_ref_host.data(), cy_ref_dev, size_in_bytes_2,
                                cudaMemcpyDeviceToHost));

  // Check y, hy, cy.
  EXPECT_EQ(y_ref_host, y_host);
  EXPECT_EQ(hy_ref_host, hy_host);
  EXPECT_EQ(cy_ref_host, cy_host);

  CUDNNXX_CUDA_CHECK(cudaFree(cy_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(hy_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(y_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(workspace));
  CUDNNXX_CUDA_CHECK(cudaFree(w_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(cy_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(hy_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(cx_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(hx_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(y_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
}

TEST_F(RNNTest, TestBackwardData) {
  Dropout<float> dropout(handle, dropout_p, seed);
  int input_n_elem = 3;
  int hidden_n_elem = input_n_elem;
  int n_layers = 1;
  RNN<float> rnn(handle, hidden_n_elem, n_layers, dropout, CUDNN_LINEAR_INPUT,
                 CUDNN_UNIDIRECTIONAL, CUDNN_RNN_RELU, CUDNN_RNN_ALGO_STANDARD,
                 dtype);

  int n_dims = 3;
  int batch_n_elem = 2;
  std::vector<int> dims_1 = {batch_n_elem, input_n_elem, 1};
  std::vector<int> strides_1 = {dims_1[2] * dims_1[1], dims_1[2], 1};
  int seq_length = 3;
  int n_elem_1 = dims_1[0] * dims_1[1] * dims_1[2] * seq_length;
  size_t size_in_bytes_1 = sizeof(float) * n_elem_1;

  // x
  std::vector<float> x_host = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                               1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8};
  float* x_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, size_in_bytes_1));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host.data(), size_in_bytes_1,
                                cudaMemcpyHostToDevice));
  TensorArray<float> x_tensors(dtype, n_dims, dims_1.data(), strides_1.data(),
                               x_dev, seq_length);

  // y
  float* y_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_dev, size_in_bytes_1));
  TensorArray<float> y_tensors(dtype, n_dims, dims_1.data(), strides_1.data(),
                               y_dev, seq_length);

  std::vector<int> dims_2 = {n_layers, batch_n_elem, hidden_n_elem};
  std::vector<int> strides_2 = {dims_2[2] * dims_2[1], dims_2[2], 1};
  int n_elem_2 = dims_2[0] * dims_2[1] * dims_2[2];
  size_t size_in_bytes_2 = sizeof(float) * n_elem_2;

  // hx
  std::vector<float> hx_host = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  float* hx_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&hx_dev, size_in_bytes_2));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(hx_dev, hx_host.data(), size_in_bytes_2,
                                cudaMemcpyHostToDevice));
  Tensor<float> hx_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                          hx_dev);

  // cx
  std::vector<float> cx_host = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  float* cx_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&cx_dev, size_in_bytes_2));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(cx_dev, cx_host.data(), size_in_bytes_2,
                                cudaMemcpyHostToDevice));
  Tensor<float> cx_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                          cx_dev);

  // hy
  float* hy_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&hy_dev, size_in_bytes_2));
  Tensor<float> hy_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                          hy_dev);

  // cy
  float* cy_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&cy_dev, size_in_bytes_2));
  Tensor<float> cy_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                          cy_dev);

  // w
  auto w_size_in_bytes = rnn.GetParamsSize(handle, x_tensors, dtype);
  std::vector<int> w_dims = {static_cast<int>(w_size_in_bytes / sizeof(float)),
                             1, 1};
  std::vector<float> w_host(w_dims[0] * w_dims[1] * w_dims[2]);
  w_host = {1};
  float* w_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&w_dev, w_size_in_bytes));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(w_dev, w_host.data(), w_size_in_bytes,
                                cudaMemcpyHostToDevice));
  Filter<float> w_filter(dtype, CUDNN_TENSOR_NCHW, n_dims, w_dims.data(),
                         w_dev);

  // workspace
  auto workspace_size_in_bytes =
      rnn.GetWorkspaceSize(handle, seq_length, x_tensors);
  void* workspace = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&workspace, workspace_size_in_bytes));

  // reserve_space
  auto reserve_size_in_bytes =
      rnn.GetTrainingReserveSize(handle, seq_length, x_tensors);
  void* reserve_space = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&reserve_space, reserve_size_in_bytes));

  // Compute.
  rnn.ForwardTraining(handle, seq_length, x_tensors, hx_tensor, cx_tensor,
                      w_filter, &y_tensors, &hy_tensor, &cy_tensor, workspace,
                      workspace_size_in_bytes, reserve_space,
                      reserve_size_in_bytes);

  // y_host
  std::vector<float> y_host(n_elem_1);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(y_host.data(), y_dev, size_in_bytes_1,
                                cudaMemcpyDeviceToHost));

  // hy_host
  std::vector<float> hy_host(n_elem_2);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(hy_host.data(), hy_dev, size_in_bytes_2,
                                cudaMemcpyDeviceToHost));

  // cy_host
  std::vector<float> cy_host(n_elem_2);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(cy_host.data(), cy_dev, size_in_bytes_2,
                                cudaMemcpyDeviceToHost));

  // y_ref
  float* y_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_ref_dev, size_in_bytes_1));
  TensorArray<float> y_ref_tensors(dtype, n_dims, dims_1.data(),
                                   strides_1.data(), y_ref_dev, seq_length);

  // hy_ref
  float* hy_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&hy_ref_dev, size_in_bytes_2));
  Tensor<float> hy_ref_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                              hy_ref_dev);

  // cy_ref
  float* cy_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&cy_ref_dev, size_in_bytes_2));
  Tensor<float> cy_ref_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                              cy_ref_dev);

  // Compute reference.
  CUDNNXX_DNN_CHECK(cudnnRNNForwardTraining(
      handle.raw_handle(), rnn.desc(), seq_length, x_tensors.descs(),
      x_tensors.dev_mem(), hx_tensor.desc(), hx_tensor.dev_mem(),
      cx_tensor.desc(), cx_tensor.dev_mem(), w_filter.desc(),
      w_filter.dev_mem(), y_ref_tensors.descs(), y_ref_tensors.dev_mem(),
      hy_ref_tensor.desc(), hy_ref_tensor.dev_mem(), cy_ref_tensor.desc(),
      cy_ref_tensor.dev_mem(), workspace, workspace_size_in_bytes,
      reserve_space, reserve_size_in_bytes));

  // y_ref_host
  std::vector<float> y_ref_host(n_elem_1);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(y_ref_host.data(), y_ref_dev, size_in_bytes_1,
                                cudaMemcpyDeviceToHost));

  // hy_ref_host
  std::vector<float> hy_ref_host(n_elem_2);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(hy_ref_host.data(), hy_ref_dev, size_in_bytes_2,
                                cudaMemcpyDeviceToHost));

  // cy_ref_host
  std::vector<float> cy_ref_host(n_elem_2);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(cy_ref_host.data(), cy_ref_dev, size_in_bytes_2,
                                cudaMemcpyDeviceToHost));

  // Check y, hy, cy.
  EXPECT_EQ(y_ref_host, y_host);
  EXPECT_EQ(hy_ref_host, hy_host);
  EXPECT_EQ(cy_ref_host, cy_host);

  // dx
  float* dx_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dx_dev, size_in_bytes_1));
  TensorArray<float> dx_tensors(dtype, n_dims, dims_1.data(), strides_1.data(),
                                dx_dev, seq_length);

  // dhx
  float* dhx_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dhx_dev, size_in_bytes_2));
  Tensor<float> dhx_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                           dhx_dev);

  // dcx
  float* dcx_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dcx_dev, size_in_bytes_2));
  Tensor<float> dcx_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                           dcx_dev);

  // Compute backward data.
  rnn.BackwardData(handle, seq_length, y_tensors, y_tensors, hy_tensor,
                   cy_tensor, w_filter, hx_tensor, cx_tensor, &dx_tensors,
                   &dhx_tensor, &dcx_tensor, workspace, workspace_size_in_bytes,
                   reserve_space, reserve_size_in_bytes);

  // dx_host
  std::vector<float> dx_host(n_elem_1);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(dx_host.data(), dx_dev, size_in_bytes_1,
                                cudaMemcpyDeviceToHost));

  // dhx_host
  std::vector<float> dhx_host(n_elem_2);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(dhx_host.data(), dhx_dev, size_in_bytes_2,
                                cudaMemcpyDeviceToHost));

  // dcx_host
  std::vector<float> dcx_host(n_elem_2);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(dcx_host.data(), dcx_dev, size_in_bytes_2,
                                cudaMemcpyDeviceToHost));

  // dx_ref
  float* dx_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dx_ref_dev, size_in_bytes_1));
  TensorArray<float> dx_ref_tensors(dtype, n_dims, dims_1.data(),
                                    strides_1.data(), dx_ref_dev, seq_length);

  // dhx
  float* dhx_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dhx_ref_dev, size_in_bytes_2));
  Tensor<float> dhx_ref_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                               dhx_ref_dev);

  // dcx
  float* dcx_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dcx_ref_dev, size_in_bytes_2));
  Tensor<float> dcx_ref_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                               dcx_ref_dev);

  // Compute backward data reference.
  CUDNNXX_DNN_CHECK(cudnnRNNBackwardData(
      handle.raw_handle(), rnn.desc(), seq_length, y_tensors.descs(),
      y_tensors.dev_mem(), y_tensors.descs(), y_tensors.dev_mem(),
      hy_tensor.desc(), hy_tensor.dev_mem(), cy_tensor.desc(),
      cy_tensor.dev_mem(), w_filter.desc(), w_filter.dev_mem(),
      hx_tensor.desc(), hx_tensor.dev_mem(), cx_tensor.desc(),
      cx_tensor.dev_mem(), dx_ref_tensors.descs(), dx_ref_tensors.dev_mem(),
      dhx_ref_tensor.desc(), dhx_ref_tensor.dev_mem(), dcx_ref_tensor.desc(),
      dcx_ref_tensor.dev_mem(), workspace, workspace_size_in_bytes,
      reserve_space, reserve_size_in_bytes));

  // dx_ref_host
  std::vector<float> dx_ref_host(n_elem_1);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(dx_ref_host.data(), dx_ref_dev, size_in_bytes_1,
                                cudaMemcpyDeviceToHost));

  // dhx_ref_host
  std::vector<float> dhx_ref_host(n_elem_2);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(dhx_ref_host.data(), dhx_ref_dev,
                                size_in_bytes_2, cudaMemcpyDeviceToHost));

  // dcx_ref_host
  std::vector<float> dcx_ref_host(n_elem_2);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(dcx_ref_host.data(), dcx_ref_dev,
                                size_in_bytes_2, cudaMemcpyDeviceToHost));

  // Check dx, dhx, dcx.
  EXPECT_EQ(dx_ref_host, dx_host);
  EXPECT_EQ(dhx_ref_host, dhx_host);
  EXPECT_EQ(dcx_ref_host, dcx_host);

  CUDNNXX_CUDA_CHECK(cudaFree(dcx_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dhx_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dx_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dcx_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dhx_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dx_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(cy_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(hy_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(y_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(reserve_space));
  CUDNNXX_CUDA_CHECK(cudaFree(workspace));
  CUDNNXX_CUDA_CHECK(cudaFree(w_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(cy_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(hy_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(cx_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(hx_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(y_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
}

TEST_F(RNNTest, TestBackwardWeights) {
  Dropout<float> dropout(handle, dropout_p, seed);
  int input_n_elem = 3;
  int hidden_n_elem = input_n_elem;
  int n_layers = 1;
  RNN<float> rnn(handle, hidden_n_elem, n_layers, dropout, CUDNN_LINEAR_INPUT,
                 CUDNN_UNIDIRECTIONAL, CUDNN_RNN_RELU, CUDNN_RNN_ALGO_STANDARD,
                 dtype);

  int n_dims = 3;
  int batch_n_elem = 2;
  std::vector<int> dims_1 = {batch_n_elem, input_n_elem, 1};
  std::vector<int> strides_1 = {dims_1[2] * dims_1[1], dims_1[2], 1};
  int seq_length = 3;
  int n_elem_1 = dims_1[0] * dims_1[1] * dims_1[2] * seq_length;
  size_t size_in_bytes_1 = sizeof(float) * n_elem_1;

  // x
  std::vector<float> x_host = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                               1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8};
  float* x_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, size_in_bytes_1));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host.data(), size_in_bytes_1,
                                cudaMemcpyHostToDevice));
  TensorArray<float> x_tensors(dtype, n_dims, dims_1.data(), strides_1.data(),
                               x_dev, seq_length);

  // y
  float* y_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_dev, size_in_bytes_1));
  TensorArray<float> y_tensors(dtype, n_dims, dims_1.data(), strides_1.data(),
                               y_dev, seq_length);

  std::vector<int> dims_2 = {n_layers, batch_n_elem, hidden_n_elem};
  std::vector<int> strides_2 = {dims_2[2] * dims_2[1], dims_2[2], 1};
  int n_elem_2 = dims_2[0] * dims_2[1] * dims_2[2];
  size_t size_in_bytes_2 = sizeof(float) * n_elem_2;

  // hx
  std::vector<float> hx_host = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  float* hx_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&hx_dev, size_in_bytes_2));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(hx_dev, hx_host.data(), size_in_bytes_2,
                                cudaMemcpyHostToDevice));
  Tensor<float> hx_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                          hx_dev);

  // cx
  std::vector<float> cx_host = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  float* cx_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&cx_dev, size_in_bytes_2));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(cx_dev, cx_host.data(), size_in_bytes_2,
                                cudaMemcpyHostToDevice));
  Tensor<float> cx_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                          cx_dev);

  // hy
  float* hy_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&hy_dev, size_in_bytes_2));
  Tensor<float> hy_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                          hy_dev);

  // cy
  float* cy_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&cy_dev, size_in_bytes_2));
  Tensor<float> cy_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                          cy_dev);

  // w
  auto w_size_in_bytes = rnn.GetParamsSize(handle, x_tensors, dtype);
  std::vector<int> w_dims = {static_cast<int>(w_size_in_bytes / sizeof(float)),
                             1, 1};
  int w_n_elem = w_dims[0] * w_dims[1] * w_dims[2];
  std::vector<float> w_host(w_n_elem);
  w_host = {1};
  float* w_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&w_dev, w_size_in_bytes));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(w_dev, w_host.data(), w_size_in_bytes,
                                cudaMemcpyHostToDevice));
  Filter<float> w_filter(dtype, CUDNN_TENSOR_NCHW, n_dims, w_dims.data(),
                         w_dev);

  // workspace
  auto workspace_size_in_bytes =
      rnn.GetWorkspaceSize(handle, seq_length, x_tensors);
  void* workspace = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&workspace, workspace_size_in_bytes));

  // reserve_space
  auto reserve_size_in_bytes =
      rnn.GetTrainingReserveSize(handle, seq_length, x_tensors);
  void* reserve_space = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&reserve_space, reserve_size_in_bytes));

  // Compute.
  rnn.ForwardTraining(handle, seq_length, x_tensors, hx_tensor, cx_tensor,
                      w_filter, &y_tensors, &hy_tensor, &cy_tensor, workspace,
                      workspace_size_in_bytes, reserve_space,
                      reserve_size_in_bytes);

  // y_host
  std::vector<float> y_host(n_elem_1);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(y_host.data(), y_dev, size_in_bytes_1,
                                cudaMemcpyDeviceToHost));

  // hy_host
  std::vector<float> hy_host(n_elem_2);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(hy_host.data(), hy_dev, size_in_bytes_2,
                                cudaMemcpyDeviceToHost));

  // cy_host
  std::vector<float> cy_host(n_elem_2);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(cy_host.data(), cy_dev, size_in_bytes_2,
                                cudaMemcpyDeviceToHost));

  // y_ref
  float* y_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_ref_dev, size_in_bytes_1));
  TensorArray<float> y_ref_tensors(dtype, n_dims, dims_1.data(),
                                   strides_1.data(), y_ref_dev, seq_length);

  // hy_ref
  float* hy_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&hy_ref_dev, size_in_bytes_2));
  Tensor<float> hy_ref_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                              hy_ref_dev);

  // cy_ref
  float* cy_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&cy_ref_dev, size_in_bytes_2));
  Tensor<float> cy_ref_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                              cy_ref_dev);

  // Compute reference.
  CUDNNXX_DNN_CHECK(cudnnRNNForwardTraining(
      handle.raw_handle(), rnn.desc(), seq_length, x_tensors.descs(),
      x_tensors.dev_mem(), hx_tensor.desc(), hx_tensor.dev_mem(),
      cx_tensor.desc(), cx_tensor.dev_mem(), w_filter.desc(),
      w_filter.dev_mem(), y_ref_tensors.descs(), y_ref_tensors.dev_mem(),
      hy_ref_tensor.desc(), hy_ref_tensor.dev_mem(), cy_ref_tensor.desc(),
      cy_ref_tensor.dev_mem(), workspace, workspace_size_in_bytes,
      reserve_space, reserve_size_in_bytes));

  // y_ref_host
  std::vector<float> y_ref_host(n_elem_1);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(y_ref_host.data(), y_ref_dev, size_in_bytes_1,
                                cudaMemcpyDeviceToHost));

  // hy_ref_host
  std::vector<float> hy_ref_host(n_elem_2);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(hy_ref_host.data(), hy_ref_dev, size_in_bytes_2,
                                cudaMemcpyDeviceToHost));

  // cy_ref_host
  std::vector<float> cy_ref_host(n_elem_2);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(cy_ref_host.data(), cy_ref_dev, size_in_bytes_2,
                                cudaMemcpyDeviceToHost));

  // Check y, hy, cy.
  EXPECT_EQ(y_ref_host, y_host);
  EXPECT_EQ(hy_ref_host, hy_host);
  EXPECT_EQ(cy_ref_host, cy_host);

  // dx
  float* dx_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dx_dev, size_in_bytes_1));
  TensorArray<float> dx_tensors(dtype, n_dims, dims_1.data(), strides_1.data(),
                                dx_dev, seq_length);

  // dhx
  float* dhx_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dhx_dev, size_in_bytes_2));
  Tensor<float> dhx_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                           dhx_dev);

  // dcx
  float* dcx_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dcx_dev, size_in_bytes_2));
  Tensor<float> dcx_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                           dcx_dev);

  // Compute backward data.
  rnn.BackwardData(handle, seq_length, y_tensors, y_tensors, hy_tensor,
                   cy_tensor, w_filter, hx_tensor, cx_tensor, &dx_tensors,
                   &dhx_tensor, &dcx_tensor, workspace, workspace_size_in_bytes,
                   reserve_space, reserve_size_in_bytes);

  // dx_host
  std::vector<float> dx_host(n_elem_1);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(dx_host.data(), dx_dev, size_in_bytes_1,
                                cudaMemcpyDeviceToHost));

  // dhx_host
  std::vector<float> dhx_host(n_elem_2);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(dhx_host.data(), dhx_dev, size_in_bytes_2,
                                cudaMemcpyDeviceToHost));

  // dcx_host
  std::vector<float> dcx_host(n_elem_2);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(dcx_host.data(), dcx_dev, size_in_bytes_2,
                                cudaMemcpyDeviceToHost));

  // dx_ref
  float* dx_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dx_ref_dev, size_in_bytes_1));
  TensorArray<float> dx_ref_tensors(dtype, n_dims, dims_1.data(),
                                    strides_1.data(), dx_ref_dev, seq_length);

  // dhx_ref
  float* dhx_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dhx_ref_dev, size_in_bytes_2));
  Tensor<float> dhx_ref_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                               dhx_ref_dev);

  // dcx_ref
  float* dcx_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dcx_ref_dev, size_in_bytes_2));
  Tensor<float> dcx_ref_tensor(dtype, n_dims, dims_2.data(), strides_2.data(),
                               dcx_ref_dev);

  // Compute backward data reference.
  CUDNNXX_DNN_CHECK(cudnnRNNBackwardData(
      handle.raw_handle(), rnn.desc(), seq_length, y_tensors.descs(),
      y_tensors.dev_mem(), y_tensors.descs(), y_tensors.dev_mem(),
      hy_tensor.desc(), hy_tensor.dev_mem(), cy_tensor.desc(),
      cy_tensor.dev_mem(), w_filter.desc(), w_filter.dev_mem(),
      hx_tensor.desc(), hx_tensor.dev_mem(), cx_tensor.desc(),
      cx_tensor.dev_mem(), dx_ref_tensors.descs(), dx_ref_tensors.dev_mem(),
      dhx_ref_tensor.desc(), dhx_ref_tensor.dev_mem(), dcx_ref_tensor.desc(),
      dcx_ref_tensor.dev_mem(), workspace, workspace_size_in_bytes,
      reserve_space, reserve_size_in_bytes));

  // dx_ref_host
  std::vector<float> dx_ref_host(n_elem_1);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(dx_ref_host.data(), dx_ref_dev, size_in_bytes_1,
                                cudaMemcpyDeviceToHost));

  // dhx_ref_host
  std::vector<float> dhx_ref_host(n_elem_2);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(dhx_ref_host.data(), dhx_ref_dev,
                                size_in_bytes_2, cudaMemcpyDeviceToHost));

  // dcx_ref_host
  std::vector<float> dcx_ref_host(n_elem_2);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(dcx_ref_host.data(), dcx_ref_dev,
                                size_in_bytes_2, cudaMemcpyDeviceToHost));

  // Check dx, dhx, dcx.
  EXPECT_EQ(dx_ref_host, dx_host);
  EXPECT_EQ(dhx_ref_host, dhx_host);
  EXPECT_EQ(dcx_ref_host, dcx_host);

  // dw
  float* dw_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dw_dev, w_size_in_bytes));
  Filter<float> dw_filter(dtype, CUDNN_TENSOR_NCHW, n_dims, w_dims.data(),
                          dw_dev);

  // Compute backward weights.
  rnn.BackwardWeights(handle, seq_length, x_tensors, hx_tensor, y_tensors,
                      workspace, workspace_size_in_bytes, &dw_filter,
                      reserve_space, reserve_size_in_bytes);

  // dw_host
  std::vector<float> dw_host(w_n_elem);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(dw_host.data(), dw_dev, w_size_in_bytes,
                                cudaMemcpyDeviceToHost));

  // dw_ref
  float* dw_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dw_ref_dev, w_size_in_bytes));
  Filter<float> dw_ref_filter(dtype, CUDNN_TENSOR_NCHW, n_dims, w_dims.data(),
                              dw_ref_dev);

  // Compute backward weights reference.
  CUDNNXX_DNN_CHECK(cudnnRNNBackwardWeights(
      handle.raw_handle(), rnn.desc(), seq_length, x_tensors.descs(),
      x_tensors.dev_mem(), hx_tensor.desc(), hx_tensor.dev_mem(),
      y_tensors.descs(), y_tensors.dev_mem(), workspace,
      workspace_size_in_bytes, dw_ref_filter.desc(), dw_ref_filter.dev_mem(),
      reserve_space, reserve_size_in_bytes));

  // dw_ref_host
  std::vector<float> dw_ref_host(w_n_elem);
  CUDNNXX_CUDA_CHECK(cudaMemcpy(dw_ref_host.data(), dw_ref_dev, w_size_in_bytes,
                                cudaMemcpyDeviceToHost));

  // Check dw.
  EXPECT_EQ(dw_ref_host, dw_host);

  CUDNNXX_CUDA_CHECK(cudaFree(dw_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dw_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dcx_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dhx_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dx_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dcx_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dhx_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dx_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(cy_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(hy_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(y_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(reserve_space));
  CUDNNXX_CUDA_CHECK(cudaFree(workspace));
  CUDNNXX_CUDA_CHECK(cudaFree(w_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(cy_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(hy_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(cx_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(hx_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(y_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
}
}  // namespace cudnnxx
