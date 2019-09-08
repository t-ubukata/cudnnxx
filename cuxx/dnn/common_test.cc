#include "cuxx/dnn/common_test.h"

#include <iostream>

#include "cuxx/dnn/common.h"

namespace cuxx {
namespace dnn {

void TestGetDeviceCount() {

  std::cerr << "Device count: " << dnn::GetDeviceCount() << std::endl;

}

}  // namespace dnn
}  // namespace cuxx
