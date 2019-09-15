#include <iostream>

#include "gtest/gtest.h"
#include "cuxx/dnn/common.h"

namespace cuxx {
namespace dnn {

class CommonTest : public ::testing::Test {};

TEST_F(CommonTest, GetDeviceCount) {
  std::cerr << "Device count: " << cuxx::dnn::GetDeviceCount() << std::endl;
}

}  // namespace dnn
}  // namespace cuxx
