#ifndef CUDNNXX_RNN_H_
#define CUDNNXX_RNN_H_

#include <vector>

#include "cudnn.h"
#include "cudnnxx/common.h"
#include "cudnnxx/dropout.h"
#include "cudnnxx/util.h"

namespace cudnnxx {

// FactorT must be float or double.
template <typename TensorT, typename FactorT>
class RNN {
 public:
  RNN(const Handle& handle, int hidden_size, int num_layers,
      const Dropout<TensorT>& dropout, cudnnRNNInputMode_t input_mode,
      cudnnDirectionMode_t direction, cudnnRNNMode_t mode, cudnnRNNAlgo_t algo,
      cudnnDataType_t dtype) {
    CUDNNXX_DNN_CHECK(cudnnCreateRNNDescriptor(&desc_));
    // NOTE: cuDNN documentation is wrong.
    //       This function takes 10 arguments in fact.
    CUDNNXX_DNN_CHECK(cudnnSetRNNDescriptor_v6(
        handle.raw_handle(), desc_, hidden_size, num_layers, dropout.desc(),
        input_mode, direction, mode, algo, dtype));
  }

  ~RNN() { CUDNNXX_DNN_CHECK(cudnnDestroyRNNDescriptor(desc_)); }

  RNN(const RNN&) = delete;
  RNN operator=(const RNN&) = delete;

  cudnnRNNDescriptor_t desc() const { return desc_; }

  size_t GetParamsSize(const Handle& handle, const Tensor<TensorT>& x,
                       cudnnDataType_t dtype) {
    size_t size_in_bytes = 0;
    CUDNNXX_DNN_CHECK(cudnnGetRNNParamsSize(handle.raw_handle(), desc_,
                                            x.desc(), &size_in_bytes, dtype));
    return size_in_bytes;
  }

  size_t GetTrainingReserveSize(const Handle& handle, int seq_length,
                                const std::vector<Tensor<float>>& xs) {
    std::vector<cudnnTensorDescriptor_t> x_descs;
    for (int i = 0; i < seq_length; ++i) {
      x_descs.push_back(xs[i].desc());
    }
    size_t size_in_bytes = 0;
    CUDNNXX_DNN_CHECK(cudnnGetRNNTrainingReserveSize(handle.raw_handle(), desc_,
                                                     seq_length, x_descs.data(),
                                                     &size_in_bytes));
    return size_in_bytes;
  }

  // TODO:
  // cudnnGetRNNWorkspaceSize

  // cudnnRNNForwardInference
  // cudnnRNNForwardTraining

  // cudnnRNNBackwardData
  // cudnnRNNBackwardWeights

  // cudnnCreatePersistentRNNPlan
  // cudnnDestroyPersistentRNNPlan
  // cudnnSetPersistentRNNPlan

  // cudnnFindRNNBackwardDataAlgorithmEx
  // cudnnFindRNNBackwardWeightsAlgorithmEx
  // cudnnFindRNNForwardInferenceAlgorithmEx
  // cudnnFindRNNForwardTrainingAlgorithmEx

  // cudnnGetRNNLinLayerBiasParams
  // cudnnGetRNNLinLayerMatrixParams

  // cudnnGetRNNProjectionLayers
  // cudnnSetRNNProjectionLayers

  // cudnnSetRNNMatrixMathType

 private:
  cudnnRNNDescriptor_t desc_;
};

}  // namespace cudnnxx

#endif  // CUDNNXX_RNN_H_
