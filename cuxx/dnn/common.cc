#include "cuxx/dnn/common.h"

#include <iostream>

#include "cuda_runtime.h"
#include "cudnn.h"

namespace cuxx {
namespace dnn {

int GetDeviceCount() {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  return device_count;
}

}  // namespace dnn
}  // namespace cuxx
