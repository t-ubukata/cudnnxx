#ifndef CUXX_DNN_COMMON_H_
#define CUXX_DNN_COMMON_H_

#include "cudnn.h"
#include "cuxx/util.h"

namespace cuxx {
namespace dnn {

// TODO(t-ubukata): Consider about copy constructors and move constructors.

class Handle {
 public:
  Handle() {
    CUXX_DNN_CHECK(cudnnCreate(&raw_handle_));
  }

  ~Handle() {
    CUXX_DNN_CHECK(cudnnDestroy(raw_handle_));
  }

  Handle(const Handle&) = delete;
  Handle(const Handle&&) = delete;

  cudnnHandle_t raw_handle() {return raw_handle_;}

 private:
  cudnnHandle_t raw_handle_;
};

// The caller must allocate dev_mem previously.
// TODO(t-ubukata): Consider about type constraints.
template<typename T>
class Tensor {
 public:
  Tensor(cudnnDataType_t dataType, cudnnTensorFormat_t format,
         int n, int c, int h, int w, T* dev_mem) : dev_mem_(dev_mem) {
    CUXX_DNN_CHECK(cudnnCreateTensorDescriptor(&desc_));
    CUXX_DNN_CHECK(cudnnSetTensor4dDescriptor(desc_, format, dataType,
                                              n, c, h, w));
  }

  Tensor(cudnnDataType_t dataType, int n, int c, int h, int w,
         int nStride, int cStride, int hStride, int wStride, T* dev_mem)
         : dev_mem_(dev_mem) {
    CUXX_DNN_CHECK(cudnnCreateTensorDescriptor(&desc_));
    CUXX_DNN_CHECK(cudnnSetTensor4dDescriptorEx(desc_, dataType,
                   n, c, h, w, nStride, cStride, hStride, wStride));
  }

  Tensor(cudnnDataType_t dataType, int nbDims, int dimA[], int strideA[],
         T* dev_mem) : dev_mem_(dev_mem) {
    CUXX_DNN_CHECK(cudnnCreateTensorDescriptor(&desc_));
    CUXX_DNN_CHECK(cudnnSetTensorNdDescriptor(desc_, dataType, nbDims, dimA,
                                              strideA));
  }

  Tensor(cudnnTensorFormat_t format, cudnnDataType_t dataType, int nbDims,
         int dimA[], T* dev_mem) : dev_mem_(dev_mem) {
    CUXX_DNN_CHECK(cudnnCreateTensorDescriptor(&desc_));
    CUXX_DNN_CHECK(cudnnSetTensorNdDescriptorEx(desc_, format, dataType, nbDims,
                                                dimA));
  }

  Tensor(const Tensor&) = delete;
  Tensor(const Tensor&&) = delete;

  ~Tensor() {
    CUXX_DNN_CHECK(cudnnDestroyTensorDescriptor(desc_));
  }

  cudnnTensorDescriptor_t desc() {return desc;}
  T* dev_mem() {return dev_mem_;}

 private:
  cudnnTensorDescriptor_t desc_;
  T* dev_mem_;
};

}  // namespace dnn
}  // namespace cuxx

#endif  // CUXX_DNN_COMMON_H_
