#ifndef CUDNNXX_REDUCE_TENSOR_H_
#define CUDNNXX_REDUCE_TENSOR_H_

#include "cudnn.h"
#include "cudnnxx/common.h"
#include "cudnnxx/util.h"

namespace cudnnxx {

template <typename TensorT, typename FactorT>
class ReduceTensor {
 public:
  ReduceTensor(cudnnReduceTensorOp_t op, cudnnDataType_t dtype,
               cudnnNanPropagation_t nan_opt,
               cudnnReduceTensorIndices_t reduces_indices,
               cudnnIndicesType_t indices_type) {
    CUDNNXX_DNN_CHECK(cudnnCreateReduceTensorDescriptor(&desc_));
    CUDNNXX_DNN_CHECK(cudnnSetReduceTensorDescriptor(
        desc_, op, dtype, nan_opt, reduces_indices, indices_type));
  }

  ~ReduceTensor() {
    CUDNNXX_DNN_CHECK(cudnnDestroyReduceTensorDescriptor(desc_));
  }

  ReduceTensor(const ReduceTensor&) = delete;
  ReduceTensor operator=(const ReduceTensor&) = delete;

  cudnnReduceTensorDescriptor_t desc() const { return desc_; }

  void Compute(const Handle& handle, void* indices,
               size_t indices_n_bytes, void* workspace,
               size_t workspace_n_bytes, FactorT alpha,
               const Tensor<TensorT>& a, FactorT beta,
               Tensor<TensorT>* c) const {
    CUDNNXX_DNN_CHECK(cudnnReduceTensor(
        handle.raw_handle(), desc_, indices, indices_n_bytes, workspace,
        workspace_n_bytes, &alpha, a.desc(), a.dev_mem(), &beta,
        c->desc(), c->dev_mem()));
  }

 private:
  cudnnReduceTensorDescriptor_t desc_;
};

}  // namespace cudnnxx

#endif  // CUDNNXX_REDUCE_TENSOR_H_
