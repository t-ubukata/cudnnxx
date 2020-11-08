#ifndef CUDNNXX_DIVISIVE_NORMALIZATION_H_
#define CUDNNXX_DIVISIVE_NORMALIZATION_H_

#include "cudnn.h"
#include "cudnnxx/common.h"
#include "cudnnxx/util.h"

namespace cudnnxx {

// FactorT must be float or double.
template <typename TensorT, typename FactorT>
class DivisiveNormalization {
 public:
  DivisiveNormalization(unsigned int n, double alpha, double beta, double k) {
    CUDNNXX_CHECK(n >= CUDNN_LRN_MIN_N && n <= CUDNN_LRN_MAX_N,
                  "n must be in range from " + std::to_string(CUDNN_LRN_MIN_N) +
                      " to " + std::to_string(CUDNN_LRN_MAX_N));
    CUDNNXX_CHECK(beta >= CUDNN_LRN_MIN_BETA,
                  "beta must be larger than or equal to " +
                      std::to_string(CUDNN_LRN_MIN_BETA));
    CUDNNXX_CHECK(k >= CUDNN_LRN_MIN_K,
                  "beta must be larger than or equal to " +
                      std::to_string(CUDNN_LRN_MIN_K));
    CUDNNXX_DNN_CHECK(cudnnCreateLRNDescriptor(&desc_));
    CUDNNXX_DNN_CHECK(cudnnSetLRNDescriptor(desc_, n, alpha, beta, k));
  }

  ~DivisiveNormalization() {
    CUDNNXX_DNN_CHECK(cudnnDestroyLRNDescriptor(desc_));
  }

  DivisiveNormalization(const DivisiveNormalization&) = delete;
  DivisiveNormalization operator=(const DivisiveNormalization&) = delete;

  cudnnLRNDescriptor_t desc() const { return desc_; }

  void Forward(const Handle& handle, FactorT alpha, const Tensor<TensorT>& x,
               const Tensor<TensorT>& means, const Tensor<TensorT>& temp,
               const Tensor<TensorT>& temp2, FactorT beta, Tensor<TensorT>* y) {
    CUDNNXX_DNN_CHECK(cudnnDivisiveNormalizationForward(
        handle.raw_handle(), desc_, mode_, &alpha, x.desc(), x.dev_mem(),
        means.dev_mem(), temp.dev_mem(), temp2.dev_mem(), &beta, y->desc(),
        y->dev_mem()));
  }

  void Backward(const Handle& handle, FactorT alpha, const Tensor<TensorT>& x,
                const Tensor<TensorT>& means, const Tensor<TensorT>& dy,
                const Tensor<TensorT>& temp, const Tensor<TensorT>& temp2,
                FactorT beta, Tensor<TensorT>* dx, Tensor<TensorT>* d_means) {
    CUDNNXX_DNN_CHECK(cudnnDivisiveNormalizationBackward(
        handle.raw_handle(), desc_, mode_, &alpha, x.desc(), x.dev_mem(),
        means.dev_mem(), dy.dev_mem(), temp.dev_mem(), temp2.dev_mem(), &beta,
        dx->desc(), dx->dev_mem(), d_means->dev_mem()));
  }

 private:
  cudnnLRNDescriptor_t desc_;
  cudnnDivNormMode_t mode_ = CUDNN_DIVNORM_PRECOMPUTED_MEANS;
};

}  // namespace cudnnxx

#endif  // CUDNNXX_DIVISIVE_NORMALIZATION_H_
