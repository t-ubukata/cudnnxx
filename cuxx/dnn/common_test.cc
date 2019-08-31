#include "cuxx/dnn/common.h"

#include <iostream>

namespace cuxx {
namespace dnn {

int TestGetDeviceCount() {

  std::cerr << "Device count: " << dnn::GetDeviceCount();

}

}  // namespace dnn
}  // namespace cuxx
