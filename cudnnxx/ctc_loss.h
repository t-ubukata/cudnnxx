#ifndef CUDNNXX_CTC_LOSS_H_
#define CUDNNXX_CTC_LOSS_H_

#include <vector>

#include "cudnn.h"
#include "cudnnxx/common.h"
#include "cudnnxx/util.h"

namespace cudnnxx {

// TODO: Consider to implement as one function.
template <typename TensorT>
class CTCLoss {
 public:
  CTCLoss(cudnnDataType_t comp_type) {
    CUDNNXX_DNN_CHECK(cudnnCreateCTCLossDescriptor(&desc_));
    CUDNNXX_DNN_CHECK(cudnnSetCTCLossDescriptor(desc_, comp_type));
  }

  CTCLoss(cudnnDataType_t comp_type, cudnnLossNormalizationMode_t norm_mode,
          cudnnNanPropagation_t grad_mode) {
    CUDNNXX_DNN_CHECK(cudnnCreateCTCLossDescriptor(&desc_));
    CUDNNXX_DNN_CHECK(
        cudnnSetCTCLossDescriptorEx(desc_, comp_type, norm_mode, grad_mode));
  }

  ~CTCLoss() { CUDNNXX_DNN_CHECK(cudnnDestroyCTCLossDescriptor(desc_)); }

  CTCLoss(const CTCLoss&) = delete;
  CTCLoss operator=(const CTCLoss&) = delete;

  cudnnCTCLossDescriptor_t desc() const { return desc_; }

  size_t GetWorkspaceSize(const Handle& handle, const Tensor<TensorT>& probs,
                          const Tensor<TensorT>& gradients, int* labels,
                          int* label_lengths, int* input_lengths,
                          cudnnCTCLossAlgo_t algo) {
    size_t size_in_bytes = 0;
    // Note: labels is host memory and ideally an tensor.
    CUDNNXX_DNN_CHECK(cudnnGetCTCLossWorkspaceSize(
        handle.raw_handle(), probs.desc(), gradients.desc(), labels,
        label_lengths, input_lengths, algo, desc_, &size_in_bytes));
    return size_in_bytes;
  }

  void Compute(const Handle& handle, const Tensor<TensorT>& probs, int* labels,
               int* label_lengths, int* input_lengths, void* costs,
               Tensor<TensorT>* gradients, cudnnCTCLossAlgo_t algo,
               void* workspace, size_t workspace_size_in_bytes) {
    // cuDNN documentation says workSpaceSizeInBytes is type of size_t*,
    // but actually size_t.
    CUDNNXX_DNN_CHECK(cudnnCTCLoss(
        handle.raw_handle(), probs.desc(), probs.dev_mem(), labels,
        label_lengths, input_lengths, costs, gradients->desc(),
        gradients->dev_mem(), algo, desc_, workspace, workspace_size_in_bytes));
  }

 private:
  cudnnCTCLossDescriptor_t desc_;
};

}  // namespace cudnnxx

#endif  // CUDNNXX_CTC_LOSS_H_
