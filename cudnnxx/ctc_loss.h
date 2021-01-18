#ifndef CUDNNXX_CTC_LOSS_H_
#define CUDNNXX_CTC_LOSS_H_

#include "cudnn.h"
#include "cudnnxx/common.h"
#include "cudnnxx/util.h"

namespace cudnnxx {

template <typename TensorT, typename FactorT>
class CTCLoss {
 public:
  CTCLoss(cudnnDataType_t comp_type) {
    CUDNNXX_DNN_CHECK(cudnnCreateCTCLossDescriptor(&desc_));
    CUDNNXX_DNN_CHECK(cudnnSetCTCLossDescriptor(desc_, comp_type));
  }

  CTCLoss(cudnnDataType_t comp_type, cudnnLossNormalizationMode_t norm_mode,
          cudnnNanPropagation_t grad_mode) {
    CUDNNXX_DNN_CHECK(cudnnCreateCTCLossDescriptor(&desc_));
    CUDNNXX_DNN_CHECK(cudnnSetCTCLossDescriptorEx(desc_, comp_type, norm_mode,
                                                grad_mode));
  }

  ~CTCLoss() {
    CUDNNXX_DNN_CHECK(cudnnDestroyCTCLossDescriptor(desc_));
  }

  CTCLoss(const CTCLoss&) = delete;
  CTCLoss operator=(const CTCLoss&) = delete;

  cudnnCTCLossDescriptor_t desc() const { return desc_; }

  // cudnnCTCLoss()
  // cudnnGetCTCLossWorkspaceSize()

 private:
  cudnnCTCLossDescriptor_t desc_;
};

}  // namespace cudnnxx

#endif  // CUDNNXX_CTC_LOSS_H_
