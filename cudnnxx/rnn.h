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

  size_t GetWorkspaceSize(const Handle& handle, int seq_length,
                          const std::vector<Tensor<float>>& xs) {
    std::vector<cudnnTensorDescriptor_t> x_descs;
    for (int i = 0; i < seq_length; ++i) {
      x_descs.push_back(xs[i].desc());
    }
    size_t size_in_bytes = 0;
    CUDNNXX_DNN_CHECK(cudnnGetRNNWorkspaceSize(handle.raw_handle(), desc_,
                                               seq_length, x_descs.data(),
                                               &size_in_bytes));
    return size_in_bytes;
  }

  // void ForwardTraining(const Handle& handle, int seq_length,
  //                      const std::vector<Tensor<float>>& xs,
  //                      const Tensor<TensorT>& hx, const Tensor<TensorT>& cx,
  //                      const Tensor<TensorT>& w, std::vector<Tensor<float>>*
  //                      y, Tensor<TensorT>* hy, Tensor<TensorT>* cy, void*
  //                      workspace, size_t workspace_size_in_bytes, void*
  //                      reserve_space, size_t reserve_space_size_in_bytes) {
  //   std::vector<cudnnTensorDescriptor_t> x_descs;
  //   std::vector<void*> x_dev_mems;
  //   for (int i = 0; i < seq_length; ++i) {
  //     x_descs.push_back(xs[i].desc());
  //     x_dev_memes.push_back(xs[i].dev_mem());
  //   }
  //   CUDNNXX_DNN_CHECK(cudnnRNNForwardTraining(
  //       handle.raw_handle(), desc_, seq_length, x_descs.data(), x.dev_mem(),
  //       const cudnnTensorDescriptor_t hxDesc, const void* hx,
  //       const cudnnTensorDescriptor_t cxDesc, const void* cx,
  //       const cudnnFilterDescriptor_t wDesc, const void* w,
  //       const cudnnTensorDescriptor_t* yDesc, void* y,
  //       const cudnnTensorDescriptor_t hyDesc, void* hy,
  //       const cudnnTensorDescriptor_t cyDesc, void* cy, workspace,
  //       workspace_size_in_bytes, reserve_space,
  //       reserve_space_size_in_bytes));
  // }

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
