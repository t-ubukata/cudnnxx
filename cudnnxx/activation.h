#ifndef CUDNNXX_ACTIVATION_H_
#define CUDNNXX_ACTIVATION_H_

#include "cudnn.h"
#include "cudnnxx/common.h"
#include "cudnnxx/util.h"

namespace cudnnxx {

// FactorT must be float or double.
template <typename TensorT, typename FactorT>
class Activation {
 public:
  Activation(cudnnActivationMode_t mode, cudnnNanPropagation_t nan_opt,
             double coef) {
    CUDNNXX_DNN_CHECK(cudnnCreateActivationDescriptor(&desc_));
    CUDNNXX_DNN_CHECK(cudnnSetActivationDescriptor(desc_, mode, nan_opt, coef));
  }

  Activation(cudnnActivationMode_t mode, cudnnNanPropagation_t nan_opt) {
    CUDNNXX_CHECK(mode != CUDNN_ACTIVATION_CLIPPED_RELU,
                  "CUDNN_ACTIVATION_CLIPPED_RELU requires coef.");
    CUDNNXX_CHECK(mode != CUDNN_ACTIVATION_ELU,
                  "CUDNN_ACTIVATION_ELU requires coef.");
    CUDNNXX_DNN_CHECK(cudnnCreateActivationDescriptor(&desc_));
    CUDNNXX_DNN_CHECK(cudnnSetActivationDescriptor(desc_, mode, nan_opt, 0));
  }

  ~Activation() { CUDNNXX_DNN_CHECK(cudnnDestroyActivationDescriptor(desc_)); }

  Activation(const Activation&) = delete;
  Activation operator=(const Activation&) = delete;

  cudnnActivationDescriptor_t desc() const { return desc_; }

  void Forward(const Handle& handle, FactorT alpha, const Tensor<TensorT>& x,
               FactorT beta, Tensor<TensorT>* y) {
    CUDNNXX_DNN_CHECK(cudnnActivationForward(handle.raw_handle(), desc_, &alpha,
                                             x.desc(), x.dev_mem(), &beta,
                                             y->desc(), y->dev_mem()));
  }

  void Backward(const Handle& handle, FactorT alpha, const Tensor<TensorT>& y,
                const Tensor<TensorT>& dy, const Tensor<TensorT>& x,
                FactorT beta, const Tensor<TensorT>* dx) {
    CUDNNXX_DNN_CHECK(cudnnActivationBackward(
        handle.raw_handle(), desc_, &alpha, y.desc(), y.dev_mem(), dy.desc(),
        dy.dev_mem(), x.desc(), x.dev_mem(), &beta, dx->desc(), dx->dev_mem()));
  }

 private:
  cudnnActivationDescriptor_t desc_;
};

}  // namespace cudnnxx

#endif  // CUDNNXX_ACTIVATION_H_
