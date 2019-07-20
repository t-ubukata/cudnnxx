#include "common.h"

namespace cuxx{
namespace dnn{

int device_count = 0;
cudaGetDeviceCount(&device_count);
std::cerr << device_count << std::endl;

}  // namespace dnn
}  // namespace cuxx
