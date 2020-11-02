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

  size_t GetParamsSize(const Handle& handle, const TensorArray<TensorT>& x,
                       cudnnDataType_t dtype) {
    size_t size_in_bytes = 0;
    CUDNNXX_DNN_CHECK(cudnnGetRNNParamsSize(
        handle.raw_handle(), desc_, x.descs()[0], &size_in_bytes, dtype));
    return size_in_bytes;
  }

  size_t GetTrainingReserveSize(const Handle& handle, int seq_length,
                                const TensorArray<TensorT>& x) {
    size_t size_in_bytes = 0;
    CUDNNXX_DNN_CHECK(cudnnGetRNNTrainingReserveSize(
        handle.raw_handle(), desc_, seq_length, x.descs(), &size_in_bytes));
    return size_in_bytes;
  }

  size_t GetWorkspaceSize(const Handle& handle, int seq_length,
                          const TensorArray<TensorT>& x) {
    size_t size_in_bytes = 0;
    CUDNNXX_DNN_CHECK(cudnnGetRNNWorkspaceSize(
        handle.raw_handle(), desc_, seq_length, x.descs(), &size_in_bytes));
    return size_in_bytes;
  }

  void ForwardTraining(const Handle& handle, int seq_length,
                       const TensorArray<TensorT>& x, const Tensor<TensorT>& hx,
                       const Tensor<TensorT>& cx, const Filter<TensorT>& w,
                       TensorArray<TensorT>* y, Tensor<TensorT>* hy,
                       Tensor<TensorT>* cy, void* workspace,
                       size_t workspace_size_in_bytes, void* reserve_space,
                       size_t reserve_space_size_in_bytes) {
    CUDNNXX_DNN_CHECK(cudnnRNNForwardTraining(
        handle.raw_handle(), desc_, seq_length, x.descs(), x.dev_mem(),
        hx.desc(), hx.dev_mem(), cx.desc(), cx.dev_mem(), w.desc(), w.dev_mem(),
        y->descs(), y->dev_mem(), hy->desc(), hy->dev_mem(), cy->desc(),
        cy->dev_mem(), workspace, workspace_size_in_bytes, reserve_space,
        reserve_space_size_in_bytes));
  }

  // TODO:
  // cudnnRNNForwardInference

  // cudnnRNNForwardTrainingEx
  // cudnnRNNForwardInferenceEx

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

  // cudnnGetRNNBiasMode
  // cudnnSetRNNBiasMode

  // cudnnGetRNNPaddingMode
  // cudnnSetRNNPaddingMode

 private:
  cudnnRNNDescriptor_t desc_;
};

}  // namespace cudnnxx

#endif  // CUDNNXX_RNN_H_
