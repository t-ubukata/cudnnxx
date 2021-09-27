#ifndef CUDNNXX_DROPOUT_H_
#define CUDNNXX_DROPOUT_H_

#include "cudnn.h"
#include "cudnnxx/common.h"
#include "cudnnxx/util.h"

namespace cudnnxx {

template <typename TensorT>
class Dropout {
 public:
  Dropout(const Handle& handle, float dropout_probability, unsigned long long seed) {
    CUDNNXX_DNN_CHECK(cudnnCreateDropoutDescriptor(&desc_));
    size_t states_n_bytes = 0;
    CUDNNXX_DNN_CHECK(
        cudnnDropoutGetStatesSize(handle.raw_handle(), &states_n_bytes));
    CUDNNXX_CUDA_CHECK(cudaMalloc(&states_, states_n_bytes));
    CUDNNXX_DNN_CHECK(cudnnSetDropoutDescriptor(desc_, handle.raw_handle(),
                                                dropout_probability, states_,
                                                states_n_bytes, seed));
  }

  ~Dropout() {
    CUDNNXX_DNN_CHECK(cudnnDestroyDropoutDescriptor(desc_));
    CUDNNXX_CUDA_CHECK(cudaFree(states_));
  }

  Dropout(const Dropout&) = delete;
  Dropout operator=(const Dropout&) = delete;

  cudnnDropoutDescriptor_t desc() const { return desc_; }

  void Forward(const Handle& handle, const Tensor<TensorT>& x,
               Tensor<TensorT>* y) {
    CUDNNXX_DNN_CHECK(cudnnDropoutGetReserveSpaceSize(
        x.desc(), &reserve_space_n_bytes_));
    CUDNNXX_CUDA_CHECK(
        cudaMalloc(&reserve_space_, reserve_space_n_bytes_));
    CUDNNXX_DNN_CHECK(cudnnDropoutForward(
        handle.raw_handle(), desc_, x.desc(), x.dev_mem(), y->desc(),
        y->dev_mem(), reserve_space_, reserve_space_n_bytes_));
  }

  void Backward(const Handle& handle, const Tensor<TensorT>& dy,
                const Tensor<TensorT>* dx) const {
    CUDNNXX_DNN_CHECK(cudnnDropoutBackward(
        handle.raw_handle(), desc_, dy.desc(), dy.dev_mem(), dx->desc(),
        dx->dev_mem(), reserve_space_, reserve_space_n_bytes_));
    CUDNNXX_CUDA_CHECK(cudaFree(reserve_space_));
  }

 private:
  void* states_ = nullptr;
  cudnnDropoutDescriptor_t desc_;
  void* reserve_space_ = nullptr;
  size_t reserve_space_n_bytes_ = 0;
};

}  // namespace cudnnxx

#endif  // CUDNNXX_DROPOUT_H_
