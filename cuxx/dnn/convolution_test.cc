#include "gtest/gtest.h"
#include "cuxx/dnn/convolution.h"

namespace cuxx {
namespace dnn {

class ConvolutionTest : public ::testing::Test {
 protected:
  Handle handle;
};

TEST_F(ConvolutionTest, TestConstructor2d) {
  Convolution<float, float> conv(4, 4, 2, 2, 2, 2, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
}

TEST_F(ConvolutionTest, TestConstructorNd) {
  constexpr int n = 3;
  constexpr int pads[n] = {4, 4, 4};
  constexpr int filter_strides[n] = {2, 2, 2};
  constexpr int dilations[n] = {2, 2, 2};
  Convolution<float, float> conv(n, pads, filter_strides, dilations,
                                 CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);
}

TEST_F(ConvolutionTest, TestGroupCount) {
  Convolution<float, float> conv(4, 4, 2, 2, 2, 2, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  constexpr int count = 2;
  conv.SetGroupCount(count);
  EXPECT_EQ(count, conv.GetGroupCount()) << "Group count mismatch.";
}

TEST_F(ConvolutionTest, TestMathType) {
  Convolution<float, float> conv(4, 4, 2, 2, 2, 2, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  constexpr cudnnMathType_t type = CUDNN_DEFAULT_MATH;
  conv.SetMathType(type);
  EXPECT_EQ(type, conv.GetMathType()) << "Math type mismatch.";
}

// No value check.
TEST_F(ConvolutionTest, TestGetForwardAlgorithmMaxCount) {
  Convolution<float, float> conv(4, 4, 2, 2, 2, 2, CUDNN_CONVOLUTION,
                                 CUDNN_DATA_FLOAT);
  int count = conv.GetForwardAlgorithmMaxCount(handle);
  CUXX_UNUSED_VAR(count);
  SUCCEED();
}

// No value check.
// TEST_F(ConvolutionTest, TestGetForwardAlgorithm) {
//   Convolution<float, float> conv(4, 4, 2, 2, 2, 2, CUDNN_CONVOLUTION,
//                                  CUDNN_DATA_FLOAT);
//   constexpr int n = 2;
//   constexpr int c = 3;
//   constexpr int h = 2;
//   constexpr int w = 2;
//   constexpr int n_elem = n * c * h * w;
//   float* mem_host[n_elem] = {};
//   float* mem_dev = nullptr;
//   size_t size = sizeof(float) * n_elem;
//   cudaMalloc(&mem_dev, size);
//   cudaMemcpy(mem_host, mem_dev, size, cudaMemcpyHostToDevice);
//   Tensor<float> x(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, h, w, mem_dev);
//   cudaFree(mem_dev);
//
//
//
//   // const Filter w;
//   // const Tensor y;
//
//   int requested_count = conv.GetForwardAlgorithmMaxCount(handle);
//   int returned_count;
//
//
//
//   EXPECT_EQ(requested_count, returned_count) << "Count mismatch.";
// }

}  // namespace dnn
}  // namespace cuxx
