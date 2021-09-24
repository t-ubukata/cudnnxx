#include "cudnnxx/spatial_transformer.h"

#include "gtest/gtest.h"

namespace cudnnxx {

class SpatialTransformerTest : public ::testing::Test {
 protected:
  Handle handle;
};

TEST_F(SpatialTransformerTest, TestConstructor) {
  constexpr int n_dims = 4;
  int dims[n_dims] = {2, 2, 3, 2};
  SpatialTransformer<float, float, float, float> st(CUDNN_DATA_FLOAT, n_dims,
                                                    dims);
}

TEST_F(SpatialTransformerTest, TestGridGeneratorForward) {
  constexpr int n_dims = 4;
  constexpr int n = 2;
  constexpr int c = 2;
  constexpr int h = 3;
  constexpr int w = 2;
  // TODO: It looks like cudnnSpatialTfGridGeneratorForward requires NCHW
  // implicitly.
  int dims[n_dims] = {n, c, h, w};
  SpatialTransformer<float, float, float, float> st(CUDNN_DATA_FLOAT, n_dims,
                                                    dims);

  constexpr int theta_n_elem = n * 2 * 3;
  float theta_host[theta_n_elem] = {};
  for (int i = 0; i < theta_n_elem; ++i) {
    theta_host[i] = i * 0.0001;
  }

  size_t theta_n_bytes = sizeof(float) * theta_n_elem;
  float* theta_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&theta_dev, theta_n_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(theta_dev, theta_host, theta_n_bytes, cudaMemcpyHostToDevice));

  constexpr int grid_n_elem = n * h * w * 2;
  size_t grid_n_bytes = sizeof(float) * grid_n_elem;

  float* grid_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&grid_ref_dev, grid_n_bytes));
  CUDNNXX_DNN_CHECK(cudnnSpatialTfGridGeneratorForward(
      handle.raw_handle(), st.desc(), theta_dev, grid_ref_dev));

  float* grid_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&grid_dev, grid_n_bytes));
  st.GridGeneratorForward(handle, theta_dev, grid_dev);

  float grid_ref[grid_n_elem] = {};
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(grid_ref, grid_ref_dev, grid_n_bytes, cudaMemcpyDeviceToHost));

  float grid[grid_n_elem] = {};
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(grid, grid_dev, grid_n_bytes, cudaMemcpyDeviceToHost));

  for (int i = 0; i < grid_n_elem; ++i) {
    EXPECT_NEAR(grid_ref[i], grid[i], 1e-4) << "Value does not match: " << i;
  }
  CUDNNXX_CUDA_CHECK(cudaFree(grid_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(grid_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(theta_dev));
}

TEST_F(SpatialTransformerTest, TestGridGeneratorBackward) {
  constexpr int n_dims = 4;
  constexpr int n = 2;
  constexpr int c = 2;
  constexpr int h = 3;
  constexpr int w = 2;
  int dims[n_dims] = {n, c, h, w};
  SpatialTransformer<float, float, float, float> st(CUDNN_DATA_FLOAT, n_dims,
                                                    dims);

  constexpr int theta_n_elem = n * 2 * 3;
  float theta_host[theta_n_elem] = {};
  for (int i = 0; i < theta_n_elem; ++i) {
    theta_host[i] = i * 0.0001;
  }

  size_t theta_n_bytes = sizeof(float) * theta_n_elem;
  float* theta_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&theta_dev, theta_n_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(theta_dev, theta_host, theta_n_bytes, cudaMemcpyHostToDevice));

  constexpr int dgrid_n_elem = n * h * w * 2;
  size_t dgrid_n_bytes = sizeof(float) * dgrid_n_elem;

  float* dgrid_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dgrid_dev, dgrid_n_bytes));
  st.GridGeneratorForward(handle, theta_dev, dgrid_dev);

  float* dtheta_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dtheta_ref_dev, theta_n_bytes));
  CUDNNXX_DNN_CHECK(cudnnSpatialTfGridGeneratorBackward(
      handle.raw_handle(), st.desc(), dgrid_dev, dtheta_ref_dev));

  float* dtheta_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dtheta_dev, theta_n_bytes));
  st.GridGeneratorBackward(handle, dgrid_dev, dtheta_dev);

  float dtheta_ref_host[theta_n_elem] = {};
  CUDNNXX_CUDA_CHECK(cudaMemcpy(dtheta_ref_host, dtheta_ref_dev, theta_n_bytes,
                                cudaMemcpyDeviceToHost));

  float dtheta_host[theta_n_elem] = {};
  CUDNNXX_CUDA_CHECK(cudaMemcpy(dtheta_host, dtheta_dev, theta_n_bytes,
                                cudaMemcpyDeviceToHost));

  for (int i = 0; i < theta_n_elem; ++i) {
    EXPECT_NEAR(dtheta_ref_host[i], dtheta_host[i], 1e-4)
        << "Value does not match: " << i;
  }
  CUDNNXX_CUDA_CHECK(cudaFree(dtheta_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dtheta_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dgrid_dev));
}

TEST_F(SpatialTransformerTest, TestSamplerForward) {
  constexpr int n_dims = 4;
  constexpr int n = 2;
  constexpr int c = 2;
  constexpr int h = 3;
  constexpr int w = 2;
  int dims[n_dims] = {n, c, h, w};
  SpatialTransformer<float, float, float, float> st(CUDNN_DATA_FLOAT, n_dims,
                                                    dims);

  constexpr int theta_n_elem = n * 2 * 3;
  float theta_host[theta_n_elem] = {};
  for (int i = 0; i < theta_n_elem; ++i) {
    theta_host[i] = i * 0.0001;
  }

  size_t theta_n_bytes = sizeof(float) * theta_n_elem;
  float* theta_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&theta_dev, theta_n_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(theta_dev, theta_host, theta_n_bytes, cudaMemcpyHostToDevice));

  constexpr int grid_n_elem = n * h * w * 2;
  size_t grid_n_bytes = sizeof(float) * grid_n_elem;

  float* grid_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&grid_dev, grid_n_bytes));
  st.GridGeneratorForward(handle, theta_dev, grid_dev);

  constexpr int tensor_n_elem = n * c * h * w;
  size_t tensor_n_bytes = sizeof(float) * tensor_n_elem;

  float alpha = 1;

  float x_host[tensor_n_elem] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                                 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4};
  float* x_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, tensor_n_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(x_dev, x_host, tensor_n_bytes, cudaMemcpyHostToDevice));
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w, x_dev);

  float beta = 1;

  float* y_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_dev, tensor_n_bytes));
  Tensor<float> y(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w, y_dev);

  st.SamplerForward(handle, alpha, x, grid_dev, beta, &y);

  float y_host[tensor_n_elem] = {};
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(&y_host, y_dev, tensor_n_bytes, cudaMemcpyDeviceToHost));

  float* y_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_ref_dev, tensor_n_bytes));
  Tensor<float> y_ref(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w,
                      y_ref_dev);

  CUDNNXX_DNN_CHECK(cudnnSpatialTfSamplerForward(
      handle.raw_handle(), st.desc(), &alpha, x.desc(), x.dev_mem(), grid_dev,
      &beta, y_ref.desc(), y_ref.dev_mem()));

  float y_ref_host[tensor_n_elem] = {};
  CUDNNXX_CUDA_CHECK(cudaMemcpy(&y_ref_host, y_ref_dev, tensor_n_bytes,
                                cudaMemcpyDeviceToHost));

  for (int i = 0; i < tensor_n_elem; ++i) {
    EXPECT_NEAR(y_ref_host[i], y_host[i], 1e-4)
        << "Value does not match: " << i;
  }
  CUDNNXX_CUDA_CHECK(cudaFree(y_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(y_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(grid_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(theta_dev));
}

TEST_F(SpatialTransformerTest, TestSamplerBackward) {
  constexpr int n_dims = 4;
  constexpr int n = 2;
  constexpr int c = 2;
  constexpr int h = 3;
  constexpr int w = 2;
  int dims[n_dims] = {n, c, h, w};
  SpatialTransformer<float, float, float, float> st(CUDNN_DATA_FLOAT, n_dims,
                                                    dims);

  constexpr int theta_n_elem = n * 2 * 3;
  float theta_host[theta_n_elem] = {};
  for (int i = 0; i < theta_n_elem; ++i) {
    theta_host[i] = i * 0.0001;
  }

  size_t theta_n_bytes = sizeof(float) * theta_n_elem;
  float* theta_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&theta_dev, theta_n_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(theta_dev, theta_host, theta_n_bytes, cudaMemcpyHostToDevice));

  constexpr int grid_n_elem = n * h * w * 2;
  size_t grid_n_bytes = sizeof(float) * grid_n_elem;

  float* grid_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&grid_dev, grid_n_bytes));
  st.GridGeneratorForward(handle, theta_dev, grid_dev);

  constexpr int tensor_n_elem = n * c * h * w;
  size_t tensor_n_bytes = sizeof(float) * tensor_n_elem;

  float alpha = 1;

  float x_host[tensor_n_elem] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                                 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4};
  float* x_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, tensor_n_bytes));
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(x_dev, x_host, tensor_n_bytes, cudaMemcpyHostToDevice));
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w, x_dev);

  float beta = 0;

  float* y_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&y_dev, tensor_n_bytes));
  Tensor<float> y(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w, y_dev);

  st.SamplerForward(handle, alpha, x, grid_dev, beta, &y);

  float* dx_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dx_dev, tensor_n_bytes));
  Tensor<float> dx(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w, dx_dev);

  float* dgrid_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dgrid_dev, grid_n_bytes));

  st.SamplerBackward(handle, alpha, x, beta, &dx, alpha, y, grid_dev, beta,
                     dgrid_dev);

  float dx_host[tensor_n_elem] = {};
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(&dx_host, dx_dev, tensor_n_bytes, cudaMemcpyDeviceToHost));

  float dgrid_host[grid_n_elem] = {};
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(&dgrid_host, dgrid_dev, grid_n_bytes, cudaMemcpyDeviceToHost));

  float* dx_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dx_ref_dev, tensor_n_bytes));
  Tensor<float> dx_ref(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w,
                       dx_ref_dev);

  float* dgrid_ref_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&dgrid_ref_dev, grid_n_bytes));

  CUDNNXX_DNN_CHECK(cudnnSpatialTfSamplerBackward(
      handle.raw_handle(), st.desc(), &alpha, x.desc(), x.dev_mem(), &beta,
      dx_ref.desc(), dx_ref.dev_mem(), &alpha, y.desc(), y.dev_mem(), grid_dev,
      &beta, dgrid_ref_dev));

  float dx_ref_host[tensor_n_elem] = {};
  CUDNNXX_CUDA_CHECK(cudaMemcpy(&dx_ref_host, dx_ref_dev, tensor_n_bytes,
                                cudaMemcpyDeviceToHost));

  float dgrid_ref_host[grid_n_elem] = {};
  CUDNNXX_CUDA_CHECK(cudaMemcpy(&dgrid_ref_host, dgrid_ref_dev, grid_n_bytes,
                                cudaMemcpyDeviceToHost));

  for (int i = 0; i < tensor_n_elem; ++i) {
    EXPECT_NEAR(dx_ref_host[i], dx_host[i], 1e-4)
        << "Value does not match: " << i;
  }
  for (int i = 0; i < grid_n_elem; ++i) {
    EXPECT_NEAR(dgrid_ref_host[i], dgrid_host[i], 1e-4)
        << "Value does not match: " << i;
  }

  CUDNNXX_CUDA_CHECK(cudaFree(dgrid_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(dx_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(y_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(grid_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(theta_dev));
}

}  // namespace cudnnxx
