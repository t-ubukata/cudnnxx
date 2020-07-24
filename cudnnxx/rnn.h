#ifndef CUDNNXX_RNN_H_
#define CUDNNXX_RNN_H_

#include "cudnn.h"
#include "cudnnxx/common.h"
#include "cudnnxx/util.h"

namespace cudnnxx {

// FactorT must be float or double.
template <typename TensorT, typename FactorT>
class RNN {
 public:
  RNN(const Handle& handle, int hidden_size, int num_layers, const Dropout<TensorT>& dropout,
      cudnnRNNInputMode_t input_mode, cudnnDirectionMode_t direction,
      cudnnRNNMode_t mode, cudnnRNNAlgo_t algo, cudnnDataType_t dtype) {
    CUDNNXX_DNN_CHECK(cudnnCreateRNNDescriptor(&desc_));
    // NOTE: cuDNN documentation is wrong.
    //       This function takes 10 arguments in fact.
    CUDNNXX_DNN_CHECK(cudnnSetRNNDescriptor_v6(handle.raw_handle(), desc_,
                                               hidden_size, num_layers,
                                               dropout.desc(), input_mode,
                                               direction, mode, algo, dtype));
  }

  ~RNN() { CUDNNXX_DNN_CHECK(cudnnDestroyRNNDescriptor(desc_)); }

  RNN(const RNN&) = delete;
  RNN operator=(const RNN&) = delete;

  cudnnRNNDescriptor_t desc() const { return desc_; }

  // void Forward(const Handle& handle, FactorT alpha, const Tensor<TensorT>& x,
  //              FactorT beta, Tensor<TensorT>* y) {
  //   CUDNNXX_DNN_CHECK(cudnnActivationForward(handle.raw_handle(), desc_,
  //   &alpha,
  //                                            x.desc(), x.dev_mem(), &beta,
  //                                            y->desc(), y->dev_mem()));
  // }
  //
  // void Backward(const Handle& handle, FactorT alpha, const Tensor<TensorT>&
  // y,
  //               const Tensor<TensorT>& dy, const Tensor<TensorT>& x,
  //               FactorT beta, const Tensor<TensorT>* dx) {
  //   CUDNNXX_DNN_CHECK(cudnnActivationBackward(
  //       handle.raw_handle(), desc_, &alpha, y.desc(), y.dev_mem(), dy.desc(),
  //       dy.dev_mem(), x.desc(), x.dev_mem(), &beta, dx->desc(),
  //       dx->dev_mem()));
  // }

  size_t GetParamsSize(const Handle& handle, const Tensor<TensorT>& x, cudnnDataType_t dtype) {
    size_t size_in_bytes = 0;
    CUDNNXX_DNN_CHECK(cudnnGetRNNParamsSize(handle.raw_handle(), desc_, x.desc(), &size_in_bytes, dtype));
    return size_in_bytes;
  }

  // TODO:
  // cudnnGetRNNTrainingReserveSize
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
