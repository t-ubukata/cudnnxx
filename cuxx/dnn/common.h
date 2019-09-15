#ifndef CUXX_DNN_COMMON_H_
#define CUXX_DNN_COMMON_H_

#include "cudnn.h"
#include "cuxx/util.h"

namespace cuxx {
namespace dnn {

class Handle {
 public:
  Handle() {
    CUXX_DNN_CHECK(cudnnCreate(&raw_handle_));
  }

  ~Handle() {
    CUXX_DNN_CHECK(cudnnDestroy(raw_handle_));
  }

  cudnnHandle_t raw_handle() {return raw_handle_;}

 private:
  cudnnHandle_t raw_handle_;
};

}  // namespace dnn
}  // namespace cuxx

#endif  // CUXX_DNN_COMMON_H_
