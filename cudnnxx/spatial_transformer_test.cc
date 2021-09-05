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
  SpatialTransformer<float, float> st(CUDNN_DATA_FLOAT, n_dims, dims);
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
  SpatialTransformer<float, float> st(CUDNN_DATA_FLOAT, n_dims, dims);

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
  // TODO: It looks like cudnnSpatialTfGridGeneratorBackward requires NCHW
  // implicitly.
  int dims[n_dims] = {n, c, h, w};
  SpatialTransformer<float, float> st(CUDNN_DATA_FLOAT, n_dims, dims);

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
}  // namespace cudnnxx
