#ifndef CUDNNXX_DROPOUT_H_
#define CUDNNXX_DROPOUT_H_

#include "cudnn.h"
#include "cudnnxx/common.h"
#include "cudnnxx/util.h"

namespace cudnnxx {

// FactorT must be float or double.
template <typename TensorT, typename FactorT>
class Dropout {
 public:
  Dropout(const Handle& handle, float dropout, unsigned long long seed) {
    CUDNNXX_DNN_CHECK(cudnnCreateDropoutDescriptor(&desc_));
    size_t state_size_in_bytes;
    CUDNNXX_DNN_CHECK(cudnnDropoutGetStatesSize(handle, state_size_in_bytes));

    // TODO: Allocate states.
    void* states;

    CUDNNXX_DNN_CHECK(cudnnSetDropoutDescriptor(desc_, handle, dropout, states,
                      state_size_in_bytes, seed));
  }

  ~Dropout() { CUDNNXX_DNN_CHECK(cudnnDestroyRNNDescriptor(desc_)); }

  Dropout(const Dropout&) = delete;
  Dropout operator=(const Dropout&) = delete;

  cudnnDropoutDescriptor_t desc() const { return desc_; }

  // void Forward(const Handle& handle, FactorT alpha, const Tensor<TensorT>& x,
  //              FactorT beta, Tensor<TensorT>* y) {
  //   CUDNNXX_DNN_CHECK(cudnnActivationForward(handle.raw_handle(), desc_, &alpha,
  //                                            x.desc(), x.dev_mem(), &beta,
  //                                            y->desc(), y->dev_mem()));
  // }
  //
  // void Backward(const Handle& handle, FactorT alpha, const Tensor<TensorT>& y,
  //               const Tensor<TensorT>& dy, const Tensor<TensorT>& x,
  //               FactorT beta, const Tensor<TensorT>* dx) {
  //   CUDNNXX_DNN_CHECK(cudnnActivationBackward(
  //       handle.raw_handle(), desc_, &alpha, y.desc(), y.dev_mem(), dy.desc(),
  //       dy.dev_mem(), x.desc(), x.dev_mem(), &beta, dx->desc(), dx->dev_mem()));
  // }

 private:
  cudnnDropoutDescriptor_t desc_;
};

}  // namespace cudnnxx

#endif  // CUDNNXX_DROPOUT_H_
