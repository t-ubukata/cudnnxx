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
  constexpr int array_length = 3;
  constexpr int pads[array_length] = {4, 4, 4};
  constexpr int filter_strides[array_length] = {2, 2, 2};
  constexpr int dilations[array_length] = {2, 2, 2};
  Convolution<float, float> conv(array_length, pads, filter_strides, dilations,
                                 CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);
}

}  // namespace dnn
}  // namespace cuxx
