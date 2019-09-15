#include <iostream>

#include "gtest/gtest.h"
#include "cuxx/dnn/common.h"

namespace cuxx {
namespace dnn {


class HandleTest : public ::testing::Test {};

TEST_F(HandleTest, Constructor) {
  Handle handle;
}

TEST_F(HandleTest, RawHandle) {
  Handle handle;
  cudnnHandle_t raw_handle = handle.raw_handle();
}

}  // namespace dnn
}  // namespace cuxx
