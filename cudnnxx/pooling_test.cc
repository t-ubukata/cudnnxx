#include "gtest/gtest.h"
#include "cudnnxx/pooling.h"

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
                             CUDNN_PROPAGATE_NAN, n_dims, window_dims,
                             paddings, strides);
}

TEST_F(PoolingTest, TestGet2dForwardOutputDim) {
  Pooling<float, float> pool(CUDNN_POOLING_MAX_DETERMINISTIC,
                             CUDNN_PROPAGATE_NAN, 2, 2, 1, 1, 2, 2);
  constexpr int x_n = 32;
  constexpr int x_c = 3;
  constexpr int x_h = 8;
  constexpr int x_w = 8;
  constexpr int n_x_elem = x_n * x_c * x_h * x_w;
  size_t x_size = sizeof(float) * n_x_elem;
  float x_host[n_x_elem] = {};
  for (int i = 0; i < n_x_elem; ++i) {
    x_host[i] = i * 0.0001;
  }
  float* x_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, x_size));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host, x_size, cudaMemcpyHostToDevice));
  Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, x_n, x_c, x_h, x_w,
                  x_dev);
  std::array<int, 4> out_dims = pool.Get2dForwardOutputDim(x);
  int out_n_ref = 0;
  int out_c_ref = 0;
  int out_h_ref = 0;
  int out_w_ref = 0;
  CUDNNXX_DNN_CHECK(cudnnGetPooling2dForwardOutputDim(pool.desc(), x.desc(),
                                                      &out_n_ref, &out_c_ref,
                                                      &out_h_ref, &out_w_ref));
  EXPECT_EQ(out_n_ref, out_dims[0]) << "Value does not match: n";
  EXPECT_EQ(out_c_ref, out_dims[1]) << "Value does not match: c";
  EXPECT_EQ(out_h_ref, out_dims[2]) << "Value does not match: h";
  EXPECT_EQ(out_w_ref, out_dims[3]) << "Value does not match: w";
  CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
}

// TODO: cudnnGetPoolingNdForwardOutputDim returns CUDNN_STATUS_BAD_PARAM.
// TEST_F(PoolingTest, TestGetNdForwardOutputDim) {
//   constexpr int n_dims = 5;
//   int window_dims[n_dims] = {1, 1, 2, 2, 2};
//   int paddings[n_dims] = {1, 1, 1, 1, 1};
//   int strides[n_dims] = {1, 1, 2, 2, 2};
//   Pooling<float, float> pool(CUDNN_POOLING_MAX_DETERMINISTIC,
//                              CUDNN_PROPAGATE_NAN, n_dims, window_dims, paddings,
//                              strides);
//   constexpr int x_n = 32;
//   constexpr int x_c = 3;
//   constexpr int x_h = 8;
//   constexpr int x_w = 8;
//   constexpr int x_d = 8;  // depth
//   constexpr int n_x_elem = x_n * x_c * x_h * x_w * x_d;
//   int dims[] = {x_n, x_c, x_h, x_w, x_d};
//   size_t x_size = sizeof(float) * n_x_elem;
//   float x_host[n_x_elem] = {};
//   for (int i = 0; i < n_x_elem; ++i) {
//     x_host[i] = i * 0.0001;
//   }
//   float* x_dev = nullptr;
//   CUDNNXX_CUDA_CHECK(cudaMalloc(&x_dev, x_size));
//   CUDNNXX_CUDA_CHECK(cudaMemcpy(x_dev, x_host, x_size, cudaMemcpyHostToDevice));
//   // Tensor<float> x(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_dims, dims, x_dev);
//   // int x_strides[] = {0, 0, 0, 0};
//   Tensor<float> x(CUDNN_DATA_FLOAT, n_dims, dims, strides, x_dev);
//   std::vector<int> out_dims = pool.GetNdForwardOutputDim(x, n_dims);
//   std::vector<int> out_dims_ref(n_dims);
//   CUDNNXX_DNN_CHECK(cudnnGetPoolingNdForwardOutputDim(pool.desc(), x.desc(),
//                                                       n_dims,
//                                                       out_dims_ref.data()));
//   for (size_t i = 0; i < out_dims.size(); ++i) {
//     EXPECT_EQ(out_dims_ref[i], out_dims[i]) << "Value does not match: " << i;
//   }
//   CUDNNXX_CUDA_CHECK(cudaFree(x_dev));
// }

}  // namespace cudnnxx
