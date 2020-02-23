#ifndef CUDNNXX_OP_TENSOR_H_
#define CUDNNXX_OP_TENSOR_H_

#include "cudnn.h"
#include "cudnnxx/common.h"
#include "cudnnxx/util.h"

namespace cudnnxx {

// FactorT must be float or double.
template <typename TensorT, typename FactorT>
class OpTensor {
 public:
  OpTensor(cudnnOpTensorOp_t op, cudnnDataType_t dtype,
           cudnnNanPropagation_t nan_opt) {
    CUDNNXX_DNN_CHECK(cudnnCreateOpTensorDescriptor(&desc_));
    CUDNNXX_DNN_CHECK(cudnnSetOpTensorDescriptor(desc_, op, dtype, nan_opt));
  }

  ~OpTensor() { CUDNNXX_DNN_CHECK(cudnnDestroyOpTensorDescriptor(desc_)); }

  OpTensor(const OpTensor&) = delete;
  OpTensor operator=(const OpTensor&) = delete;

  cudnnOpTensorDescriptor_t desc() const { return desc_; }

  void Compute(const Handle& handle, FactorT alpha1, const Tensor<TensorT>& a,
               FactorT alpha2, const Tensor<TensorT>& b, FactorT beta,
               Tensor<TensorT>* c) const {
    CUDNNXX_DNN_CHECK(cudnnOpTensor(
        handle.raw_handle(), desc_, &alpha1, a.desc(), a.dev_mem(), &alpha2,
        b.desc(), b.dev_mem(), &beta, c->desc(), c->dev_mem()));
  }

 private:
  cudnnOpTensorDescriptor_t desc_;
};

}  // namespace cudnnxx

#endif  // CUDNNXX_OP_TENSOR_H_
