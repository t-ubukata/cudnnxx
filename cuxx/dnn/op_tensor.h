#ifndef CUXX_DNN_OP_TENSOR_H_
#define CUXX_DNN_OP_TENSOR_H_

#include "cudnn.h"
#include "cuxx/util.h"

#include "cuxx/dnn/common.h"

namespace cuxx {
namespace dnn {

// TODO(t-ubukata): Consider to check types via type_traits.

// T must be float or double.
template<typename TensorT, typename FactorT>
class OpTensor {
 public:
  OpTensor(cudnnOpTensorOp_t op, cudnnDataType_t CompType,
           cudnnNanPropagation_t NanOpt) {
    CUXX_DNN_CHECK(cudnnCreateOpTensorDescriptor(&desc_));
    CUXX_DNN_CHECK(cudnnSetOpTensorDescriptor(desc_, op, CompType, NanOpt));
  }

  ~OpTensor() {
    CUXX_DNN_CHECK(cudnnDestroyOpTensorDescriptor(desc_));
  }

  OpTensor(const OpTensor&) = delete;
  OpTensor operator=(const OpTensor&) = delete;

  cudnnOpTensorDescriptor_t desc() {return desc_;}

  void Compute(const Handle& handle,
               FactorT alpha1, const Tensor<TensorT>& a,
               FactorT alpha2, const Tensor<TensorT>& b,
               FactorT beta, const Tensor<TensorT>& c) {
    const auto alpha1_lvalue = alpha1;
    const auto alpha2_lvalue = alpha2;
    const auto beta_lvalue = beta;
    CUXX_DNN_CHECK(cudnnOpTensor(handle.raw_handle(), desc_,
                   &alpha1_lvalue, a.desc(), a.dev_mem(),
                   &alpha2_lvalue, b.desc, b.dev_mem(),
                   &beta_lvalue, c.desc(), c.dev_mem()));
  }

 private:
  cudnnOpTensorDescriptor_t desc_;
};

}  // namespace dnn
}  // namespace cuxx

#endif  // CUXX_DNN_OP_TENSOR_H_
