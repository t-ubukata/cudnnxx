#ifndef CUDNNXX_COMMON_H_
#define CUDNNXX_COMMON_H_

#include <utility>
#include <vector>

#include "cudnn.h"
#include "cudnnxx/util.h"

namespace cudnnxx {

class Handle {
 public:
  Handle() { CUDNNXX_DNN_CHECK(cudnnCreate(&raw_handle_)); }

  ~Handle() { CUDNNXX_DNN_CHECK(cudnnDestroy(raw_handle_)); }

  Handle(const Handle&) = delete;
  Handle operator=(const Handle&) = delete;

  cudnnHandle_t raw_handle() const { return raw_handle_; }

 private:
  cudnnHandle_t raw_handle_;
};

// The caller must allocate dev_mem previously.
template <typename T>
class Tensor {
 public:
  Tensor(cudnnDataType_t dtype, cudnnTensorFormat_t format, int n, int c, int h,
         int w, T* dev_mem)
      : dev_mem_(dev_mem) {
    CUDNNXX_DNN_CHECK(cudnnCreateTensorDescriptor(&desc_));
    CUDNNXX_DNN_CHECK(
        cudnnSetTensor4dDescriptor(desc_, format, dtype, n, c, h, w));
  }

  Tensor(cudnnDataType_t dtype, int n, int c, int h, int w, int n_stride,
         int c_stride, int h_stride, int w_stride, T* dev_mem)
      : dev_mem_(dev_mem) {
    CUDNNXX_DNN_CHECK(cudnnCreateTensorDescriptor(&desc_));
    CUDNNXX_DNN_CHECK(cudnnSetTensor4dDescriptorEx(
        desc_, dtype, n, c, h, w, n_stride, c_stride, h_stride, w_stride));
  }

  Tensor(cudnnDataType_t dtype, int n_dims, int dims[], int strides[],
         T* dev_mem)
      : dev_mem_(dev_mem) {
    CUDNNXX_DNN_CHECK(cudnnCreateTensorDescriptor(&desc_));
    CUDNNXX_DNN_CHECK(
        cudnnSetTensorNdDescriptor(desc_, dtype, n_dims, dims, strides));
  }

  Tensor(cudnnTensorFormat_t format, cudnnDataType_t dtype, int n_dims,
         int dims[], T* dev_mem)
      : dev_mem_(dev_mem) {
    CUDNNXX_DNN_CHECK(cudnnCreateTensorDescriptor(&desc_));
    CUDNNXX_DNN_CHECK(
        cudnnSetTensorNdDescriptorEx(desc_, format, dtype, n_dims, dims));
  }

  Tensor(const Tensor&) = delete;
  Tensor operator=(const Tensor&) = delete;

  Tensor(Tensor&& rhs) noexcept
      : desc_(std::move(rhs.desc_)), dev_mem_(std::move(rhs.dev_mem_)) {
    rhs.desc_ = nullptr;
    rhs.dev_mem_ = nullptr;
  };

  ~Tensor() { CUDNNXX_DNN_CHECK(cudnnDestroyTensorDescriptor(desc_)); }

  cudnnTensorDescriptor_t desc() const { return desc_; }
  T* dev_mem() const { return dev_mem_; }

 private:
  cudnnTensorDescriptor_t desc_;
  T* dev_mem_;
};

// The caller must allocate dev_mem previously.
template <typename T>
class Filter {
 public:
  Filter(cudnnDataType_t dtype, cudnnTensorFormat_t format, int k, int c, int h,
         int w, T* dev_mem)
      : dev_mem_(dev_mem) {
    CUDNNXX_DNN_CHECK(cudnnCreateFilterDescriptor(&desc_));
    CUDNNXX_DNN_CHECK(
        cudnnSetFilter4dDescriptor(desc_, dtype, format, k, c, h, w));
  }

  Filter(cudnnDataType_t dtype, cudnnTensorFormat_t format, int n_dims,
         int dims[], T* dev_mem)
      : dev_mem_(dev_mem) {
    CUDNNXX_DNN_CHECK(cudnnCreateFilterDescriptor(&desc_));
    CUDNNXX_DNN_CHECK(
        cudnnSetFilterNdDescriptor(desc_, dtype, format, n_dims, dims));
  }

  Filter(const Filter&) = delete;
  Filter operator=(const Filter&) = delete;

  Filter(Filter&& rhs) noexcept
      : desc_(std::move(rhs.desc_)), dev_mem_(std::move(rhs.dev_mem_)) {
    rhs.desc_ = nullptr;
    rhs.dev_mem_ = nullptr;
  };

  ~Filter() { CUDNNXX_DNN_CHECK(cudnnDestroyFilterDescriptor(desc_)); }

  cudnnFilterDescriptor_t desc() const { return desc_; }
  T* dev_mem() const { return dev_mem_; }

 private:
  cudnnFilterDescriptor_t desc_;
  T* dev_mem_;
};

// The caller must allocate dev_mem previously.
template <typename T>
class TensorArray {
 public:
  TensorArray(cudnnDataType_t dtype, int n_dims, int dims[], int strides[],
              T* dev_mem, int n_tensors)
      : dev_mem_(dev_mem), n_tensors_(n_tensors) {
    for (int i = 0; i < n_tensors_; ++i) {
      cudnnTensorDescriptor_t desc;
      CUDNNXX_DNN_CHECK(cudnnCreateTensorDescriptor(&desc));
      CUDNNXX_DNN_CHECK(
          cudnnSetTensorNdDescriptor(desc, dtype, n_dims, dims, strides));
      descs_.push_back(desc);
    }
  }

  TensorArray(const TensorArray&) = delete;
  TensorArray operator=(const TensorArray&) = delete;

  TensorArray(TensorArray&& rhs) noexcept
      : descs_(std::move(rhs.descs_)),
        dev_mem_(std::move(rhs.dev_mem_)),
        n_tensors_(rhs.n_tensors_) {
    rhs.descs_ = nullptr;
    rhs.dev_mem_ = nullptr;
  };

  ~TensorArray() {
    for (int i = 0; i < n_tensors_; ++i) {
      CUDNNXX_DNN_CHECK(cudnnDestroyTensorDescriptor(descs_[i]));
    }
  }

  std::vector<cudnnTensorDescriptor_t> descs() const { return descs_; }
  T* dev_mem() const { return dev_mem_; }

 private:
  std::vector<cudnnTensorDescriptor_t> descs_;
  T* dev_mem_;
  const int n_tensors_;
};

}  // namespace cudnnxx

#endif  // CUDNNXX_COMMON_H_
