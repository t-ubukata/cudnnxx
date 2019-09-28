#include "gtest/gtest.h"
#include "cuxx/dnn/op_tensor.h"

namespace cuxx {
namespace dnn {

class OpTensorTest : public ::testing::Test {
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

}  // namespace dnn
}  // namespace cuxx
