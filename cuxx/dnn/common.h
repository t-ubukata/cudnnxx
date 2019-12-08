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

  Handle(const Handle&) = delete;
  Handle operator=(const Handle&) = delete;

  cudnnHandle_t raw_handle() const {return raw_handle_;}

 private:
  cudnnHandle_t raw_handle_;
};

// The caller must allocate dev_mem previously.
template<typename T>
class Tensor {
 public:
  Tensor(cudnnDataType_t dtype, cudnnTensorFormat_t format,
         int n, int c, int h, int w, T* dev_mem) : dev_mem_(dev_mem) {
    CUXX_DNN_CHECK(cudnnCreateTensorDescriptor(&desc_));
    CUXX_DNN_CHECK(cudnnSetTensor4dDescriptor(desc_, format, dtype,
                                              n, c, h, w));
  }

  Tensor(cudnnDataType_t dtype, int n, int c, int h, int w,
         int n_stride, int c_stride, int h_stride, int w_stride, T* dev_mem)
         : dev_mem_(dev_mem) {
    CUXX_DNN_CHECK(cudnnCreateTensorDescriptor(&desc_));
    CUXX_DNN_CHECK(cudnnSetTensor4dDescriptorEx(desc_, dtype,
                   n, c, h, w, n_stride, c_stride, h_stride, w_stride));
  }

  Tensor(cudnnDataType_t dtype, int n_dims, int dims[], int strides[],
         T* dev_mem) : dev_mem_(dev_mem) {
    CUXX_DNN_CHECK(cudnnCreateTensorDescriptor(&desc_));
    CUXX_DNN_CHECK(cudnnSetTensorNdDescriptor(desc_, dtype, n_dims, dims,
                                              strides));
  }

  Tensor(cudnnTensorFormat_t format, cudnnDataType_t dtype, int n_dims,
         int dims[], T* dev_mem) : dev_mem_(dev_mem) {
    CUXX_DNN_CHECK(cudnnCreateTensorDescriptor(&desc_));
    CUXX_DNN_CHECK(cudnnSetTensorNdDescriptorEx(desc_, format, dtype, n_dims,
                                                dims));
  }

  Tensor(const Tensor&) = delete;
  Tensor operator=(const Tensor&) = delete;

  ~Tensor() {
    CUXX_DNN_CHECK(cudnnDestroyTensorDescriptor(desc_));
  }

  cudnnTensorDescriptor_t desc() const {return desc_;}
  T* dev_mem() const {return dev_mem_;}

 private:
  cudnnTensorDescriptor_t desc_;
  T* dev_mem_;
};

// The caller must allocate dev_mem previously.
template<typename T>
class Filter {
 public:
  Filter(cudnnDataType_t dtype, cudnnTensorFormat_t format,
         int k, int c, int h, int w, T* dev_mem) : dev_mem_(dev_mem) {
    CUXX_DNN_CHECK(cudnnCreateFilterDescriptor(&desc_));
    CUXX_DNN_CHECK(cudnnSetFilter4dDescriptor(desc_, dtype, format,
                                              k, c, h, w));
  }

  Filter(cudnnDataType_t dtype, cudnnTensorFormat_t format, int n_dims,
        int dims[], T* dev_mem) : dev_mem_(dev_mem) {
    CUXX_DNN_CHECK(cudnnCreateFilterDescriptor(&desc_));
    CUXX_DNN_CHECK(cudnnSetFilterNdDescriptor(desc_, dtype, format, n_dims,
                                              dims));
  }

  Filter(const Filter&) = delete;
  Filter operator=(const Filter&) = delete;

  ~Filter() {
    CUXX_DNN_CHECK(cudnnDestroyFilterDescriptor(desc_));
  }

  cudnnFilterDescriptor_t desc() const {return desc_;}
  T* dev_mem() const {return dev_mem_;}

 private:
  cudnnFilterDescriptor_t desc_;
  T* dev_mem_;
};


}  // namespace dnn
}  // namespace cuxx

#endif  // CUXX_DNN_COMMON_H_
