#ifndef CUDNNXX_DROPOUT_H_
#define CUDNNXX_DROPOUT_H_

#include "cudnn.h"
#include "cudnnxx/common.h"
#include "cudnnxx/util.h"

namespace cudnnxx {

template <typename TensorT>
class Dropout {
 public:
  Dropout(const Handle& handle, float dropout, unsigned long long seed) {
    CUDNNXX_DNN_CHECK(cudnnCreateDropoutDescriptor(&desc_));
    size_t states_size_in_bytes;
    CUDNNXX_DNN_CHECK(
        cudnnDropoutGetStatesSize(handle.raw_handle(), &states_size_in_bytes));
    CUDNNXX_CUDA_CHECK(cudaMalloc(&states, states_size_in_bytes));
    CUDNNXX_DNN_CHECK(cudnnSetDropoutDescriptor(desc_, handle.raw_handle(),
                                                dropout, states,
                                                states_size_in_bytes, seed));
  }

  ~Dropout() {
    CUDNNXX_DNN_CHECK(cudnnDestroyDropoutDescriptor(desc_));
    CUDNNXX_CUDA_CHECK(cudaFree(states));
  }

  Dropout(const Dropout&) = delete;
  Dropout operator=(const Dropout&) = delete;

  cudnnDropoutDescriptor_t desc() const { return desc_; }

  void Forward(const Handle& handle, const Tensor<TensorT>& x,
               Tensor<TensorT>* y) {
    CUDNNXX_DNN_CHECK(cudnnDropoutGetReserveSpaceSize(
        x.desc(), &reserve_space_size_in_bytes));
    CUDNNXX_DNN_CHECK(cudnnDropoutForward(
        handle.raw_handle(), desc_, x.desc(), x.dev_mem(), y->desc(),
        y->dev_mem(), reserve_space, reserve_space_size_in_bytes));
  }

  // void Backward(const Handle& handle, FactorT alpha, const Tensor<TensorT>&
  // y,
  //               const Tensor<TensorT>& dy, const Tensor<TensorT>& x,
  //               FactorT beta, const Tensor<TensorT>* dx) {
  //   CUDNNXX_DNN_CHECK(cudnnActivationBackward(
  //       handle.raw_handle(), desc_, &alpha, y.desc(), y.dev_mem(), dy.desc(),
  //       dy.dev_mem(), x.desc(), x.dev_mem(), &beta, dx->desc(),
  //       dx->dev_mem()));
  // }

 private:
  void* states;
  cudnnDropoutDescriptor_t desc_;
  void* reserve_space;
  size_t reserve_space_size_in_bytes;
};

}  // namespace cudnnxx

#endif  // CUDNNXX_DROPOUT_H_
