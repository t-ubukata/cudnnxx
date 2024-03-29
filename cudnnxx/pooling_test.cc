#include "cudnnxx/pooling.h"

#include "gtest/gtest.h"

namespace cudnnxx {

class PoolingTest : public ::testing::Test {
 protected:
  Handle handle;
};

TEST_F(PoolingTest, TestConstructor2d) {
  Pooling<float, float> pool(CUDNN_POOLING_MAX_DETERMINISTIC,
                             CUDNN_PROPAGATE_NAN, 2, 2, 1, 1, 2, 2);
}

TEST_F(PoolingTest, TestConstructorNd) {
  constexpr int n_dims = 3;
  int window_dims[n_dims] = {2, 2, 2};
  int paddings[n_dims] = {1, 1, 1};
  int strides[n_dims] = {2, 2, 2};
  Pooling<float, float> pool(CUDNN_POOLING_MAX_DETERMINISTIC,
                             CUDNN_PROPAGATE_NAN, n_dims, window_dims, paddings,
                             strides);
}

TEST_F(PoolingTest, TestGet2dForwardOutputDim) {
  constexpr int x_n = 32;
  constexpr int x_c = 3;
  constexpr int x_h = 8;
  constexpr int x_w = 8;
  constexpr int x_n_elem = x_n * x_c * x_h * x_w;
  size_t x_n_bytes = sizeof(float) * x_n_elem;
  float x_host[x_n_elem] = {};
  for (int i = 0; i < x_n_elem; ++i) {
    x_host[i] = i * 0.0001;
  }
  float* x_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, x_n_bytes));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host, x_n_bytes, cudaMemcpyHostToDevice));
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, x_n, x_c, x_h,
                         x_w, x_dev);
  Pooling<float, float> pool(CUDNN_POOLING_MAX_DETERMINISTIC,
                             CUDNN_PROPAGATE_NAN, 2, 2, 1, 1, 2, 2);
  std::array<int, 4> out_dims = pool.Get2dForwardOutputDim(x);
  int out_n_ref = 0;
  int out_c_ref = 0;
  int out_h_ref = 0;
  int out_w_ref = 0;
  CUDNNXX_DNN_CHECK(cudnnGetPooling2dForwardOutputDim(
      pool.desc(), x.desc(), &out_n_ref, &out_c_ref, &out_h_ref,
      &out_w_ref));
  EXPECT_EQ(out_n_ref, out_dims[0]);
  EXPECT_EQ(out_c_ref, out_dims[1]);
  EXPECT_EQ(out_h_ref, out_dims[2]);
  EXPECT_EQ(out_w_ref, out_dims[3]);
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
}

TEST_F(PoolingTest, TestGetNdForwardOutputDim) {
  constexpr int n_pooling_dims = 3;
  int window_dims[n_pooling_dims] = {2, 2, 2};
  int paddings[n_pooling_dims] = {1, 1, 1};
  int pooling_strides[n_pooling_dims] = {2, 2, 2};
  Pooling<float, float> pool(CUDNN_POOLING_MAX_DETERMINISTIC,
                             CUDNN_PROPAGATE_NAN, n_pooling_dims, window_dims,
                             paddings, pooling_strides);
  constexpr int x_n = 32;
  constexpr int x_c = 3;
  constexpr int x_h = 8;
  constexpr int x_w = 8;
  constexpr int x_d = 8;  // depth
  constexpr int x_n_dims = 5;
  constexpr int x_n_elem = x_n * x_c * x_h * x_w * x_d;
  int dims[x_n_dims] = {x_n, x_c, x_h, x_w, x_d};
  size_t x_n_bytes = sizeof(float) * x_n_elem;
  float x_host[x_n_elem] = {};
  for (int i = 0; i < x_n_elem; ++i) {
    x_host[i] = i * 0.0001;
  }
  float* x_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, x_n_bytes));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host, x_n_bytes, cudaMemcpyHostToDevice));
  int x_strides[x_n_dims] = {6144, 192, 64, 8, 1};
  Tensor<float> x(CUDNN_DATA_FLOAT, x_n_dims, dims, x_strides, x_dev);
  std::vector<int> out_dims = pool.GetNdForwardOutputDim(x, x_n_dims);
  std::vector<int> out_dims_ref(x_n_dims);
  CUDNNXX_DNN_CHECK(cudnnGetPoolingNdForwardOutputDim(
      pool.desc(), x.desc(), x_n_dims, out_dims_ref.data()));
  for (size_t i = 0; i < out_dims.size(); ++i) {
    EXPECT_EQ(out_dims_ref[i], out_dims[i]) << "at index " << i;
  }
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
}

TEST_F(PoolingTest, TestForward) {
  constexpr int x_n = 32;
  constexpr int x_c = 3;
  constexpr int x_h = 8;
  constexpr int x_w = 8;
  constexpr int x_n_elem = x_n * x_c * x_h * x_w;
  size_t x_n_bytes = sizeof(float) * x_n_elem;
  float x_host[x_n_elem] = {};
  for (int i = 0; i < x_n_elem; ++i) {
    x_host[i] = i * 0.0001;
  }
  float* x_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, x_n_bytes));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host, x_n_bytes, cudaMemcpyHostToDevice));
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, x_n, x_c, x_h, x_w, x_dev);

  Pooling<float, float> pool(CUDNN_POOLING_MAX_DETERMINISTIC,
                             CUDNN_PROPAGATE_NAN, 2, 2, 1, 1, 2, 2);
  std::array<int, 4> y_dims = pool.Get2dForwardOutputDim(x);
  int y_n_elem = y_dims[0] * y_dims[1] * y_dims[2] * y_dims[3];
  size_t y_n_bytes = sizeof(float) * y_n_elem;
  float* y_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_dev, y_n_bytes));
  Tensor<float> y(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, y_dims[0],
                         y_dims[1], y_dims[2], y_dims[3], y_dev);

  pool.Forward(handle, 1, x, 0, &y);

  float* y_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_ref_dev, y_n_bytes));
  Tensor<float> y_ref(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, y_dims[0],
                             y_dims[1], y_dims[2], y_dims[3], y_ref_dev);
  float alpha = 1;
  float beta = 0;
  CUDNNXX_DNN_CHECK(cudnnPoolingForward(
      handle.raw_handle(), pool.desc(), &alpha, x.desc(),
      x.dev_mem(), &beta, y_ref.desc(), y_ref.dev_mem()));
  float y_host[x_n_elem] = {};
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(y_host, y.dev_mem(), y_n_bytes, cudaMemcpyDeviceToHost));
  float y_ref_host[x_n_elem] = {};
  CUDNNXX_CUDA_CHECK(cudaMemcpy(y_ref_host, y_ref.dev_mem(), y_n_bytes,
                                cudaMemcpyDeviceToHost));

  for (int i = 0; i < y_n_elem; ++i) {
    EXPECT_NEAR(y_ref_host[i], y_host[i], 1e-4) << "at index " << i;
  }
  CUDNNXX_CUDA_CHECK(cudaFree(y_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(y_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
}

TEST_F(PoolingTest, TestBackward) {
  constexpr int x_n = 32;
  constexpr int x_c = 3;
  constexpr int x_h = 8;
  constexpr int x_w = 8;
  constexpr int x_n_elem = x_n * x_c * x_h * x_w;
  size_t x_n_bytes = sizeof(float) * x_n_elem;
  float x_host[x_n_elem] = {};
  for (int i = 0; i < x_n_elem; ++i) {
    x_host[i] = i * 0.0001;
  }
  float* x_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, x_n_bytes));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host, x_n_bytes, cudaMemcpyHostToDevice));
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, x_n, x_c, x_h, x_w, x_dev);

  Pooling<float, float> pool(CUDNN_POOLING_MAX_DETERMINISTIC,
                             CUDNN_PROPAGATE_NAN, 2, 2, 1, 1, 2, 2);
  std::array<int, 4> y_dims = pool.Get2dForwardOutputDim(x);
  int y_n_elem = y_dims[0] * y_dims[1] * y_dims[2] * y_dims[3];
  size_t y_n_bytes = sizeof(float) * y_n_elem;
  std::vector<float> y_host(y_n_elem);
  for (int i = 0; i < y_n_elem; ++i) {
    y_host[i] = i * 0.0002;
  }
  float* y_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_dev, y_n_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(y_dev, y_host.data(), y_n_bytes, cudaMemcpyHostToDevice));
  Tensor<float> y(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, y_dims[0],
                         y_dims[1], y_dims[2], y_dims[3], x_dev);

  std::vector<float> dy_host(y_n_elem);
  for (int i = 0; i < y_n_elem; ++i) {
    dy_host[i] = i * 0.0003;
  }
  float* dy_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dy_dev, y_n_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(dy_dev, dy_host.data(), y_n_bytes, cudaMemcpyHostToDevice));
  Tensor<float> dy(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, y_dims[0],
                          y_dims[1], y_dims[2], y_dims[3], y_dev);

  float* dx_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dx_dev, x_n_bytes));
  Tensor<float> dx(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, x_n, x_c, x_h,
                          x_w, dx_dev);

  pool.Backward(handle, 1, y, dy, x, 0, &dx);

  float* dx_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dx_ref_dev, x_n_bytes));
  Tensor<float> dx_ref(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, x_n, x_c,
                              x_h, x_w, dx_ref_dev);
  float alpha = 1;
  float beta = 0;
  CUDNNXX_DNN_CHECK(cudnnPoolingBackward(
      handle.raw_handle(), pool.desc(), &alpha, y.desc(),
      y.dev_mem(), dy.desc(), dy.dev_mem(),
      x.desc(), x.dev_mem(), &beta, dx_ref.desc(),
      dx_ref.dev_mem()));
  float dx_host[x_n_elem] = {};
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(dx_host, dx.dev_mem(), x_n_bytes, cudaMemcpyDeviceToHost));
  float dx_ref_host[x_n_elem] = {};
  CUDNNXX_CUDA_CHECK(cudaMemcpy(dx_ref_host, dx_ref.dev_mem(), x_n_bytes,
                                cudaMemcpyDeviceToHost));

  for (int i = 0; i < x_n_elem; ++i) {
    EXPECT_NEAR(dx_ref_host[i], dx_host[i], 1e-4) << "at index " << i;
  }
  CUDNNXX_CUDA_CHECK(cudaFree(dx_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dx_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dy_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(y_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
}

}  // namespace cudnnxx
