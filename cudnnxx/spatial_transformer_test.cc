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

  size_t theta_size_in_bytes = sizeof(float) * theta_n_elem;
  float* theta_dev;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&theta_dev, theta_size_in_bytes));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(theta_dev, theta_host, theta_size_in_bytes,
                                cudaMemcpyHostToDevice));

  constexpr int grid_n_elem = n * h * w * 2;
  size_t grid_size_in_bytes = sizeof(float) * grid_n_elem;

  float* grid_ref_dev;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&grid_ref_dev, grid_size_in_bytes));
  CUDNNXX_DNN_CHECK(cudnnSpatialTfGridGeneratorForward(
      handle.raw_handle(), st.desc(), theta_dev, grid_ref_dev));

  float* grid_dev;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&grid_dev, grid_size_in_bytes));
  st.GridGeneratorForward(handle, theta_dev, grid_dev);

  float grid_ref[grid_n_elem] = {};
  CUDNNXX_CUDA_CHECK(cudaMemcpy(grid_ref, grid_ref_dev, grid_size_in_bytes,
                                cudaMemcpyDeviceToHost));

  float grid[grid_n_elem] = {};
  CUDNNXX_CUDA_CHECK(
      cudaMemcpy(grid, grid_dev, grid_size_in_bytes, cudaMemcpyDeviceToHost));

  for (int i = 0; i < grid_n_elem; ++i) {
    EXPECT_EQ(grid_ref[i], grid[i]) << "Value does not match: " << i;
  }
  CUDNNXX_CUDA_CHECK(cudaFree(grid_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(grid_ref_dev));
  CUDNNXX_CUDA_CHECK(cudaFree(theta_dev));
}

TEST_F(SpatialTransformerTest, TestGridGeneratorBackward) {
  // constexpr int n_dims = 4;
  // constexpr int n = 2;
  // constexpr int h = 3;
  // constexpr int w = 2;
  // constexpr int c = 2;
  // int dims[n_dims] = {n, h, w, c};
  // SpatialTransformer<float, float> st(CUDNN_DATA_FLOAT, n_dims, dims);

  // constexpr int theta_n_elem = n_dims * 2 * 3;
  // float theta[theta_n_elem] = {
  //     0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
  // };
  // for (int i = 0; i < theta_n_elem; ++i) {
  //   theta[i] = i * 0.0001;
  // }
  //
  // constexpr int grid_n_elem = n * h * w * 2;
  //
  // float grid_ref[grid_n_elem] = {};
  // CUDNNXX_DNN_CHECK(cudnnSpatialTfGridGeneratorForward(
  //     handle.raw_handle(), st.desc(), theta, grid_ref));

  // float grid[grid_n_elem] = {};
  // st.GridGeneratorForward(handle, theta, grid);

  // constexpr int dtheta_n_elem = n_dims * 2 * 3;
  // float dtheta[dtheta_n_elem] = {};

  // st.GridGeneratorBackward(handle, grid_ref, dtheta);

  // float grid_ref[grid_n_elem] = {};
  // CUDNNXX_DNN_CHECK(cudnnSpatialTfGridGeneratorForward(
  //     handle.raw_handle(), st.desc(), theta, grid_ref));
  //
  // float grid[grid_n_elem] = {};
  // st.GridGeneratorForkward(handle, theta, grid);
  //
  // for (int i = 0; i < theta_n_elem; ++i) {
  //   EXPECT_EQ(grid_ref[i], grid[i]) << "Value does not match: " << i;
  // }
}
}  // namespace cudnnxx
