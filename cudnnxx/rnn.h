#ifndef CUDNNXX_RNN_H_
#define CUDNNXX_RNN_H_

#include <vector>

#include "cudnn.h"
#include "cudnnxx/common.h"
#include "cudnnxx/dropout.h"
#include "cudnnxx/util.h"

namespace cudnnxx {

template <typename TensorT>
class RNN {
 public:
  RNN(const Handle& handle, int hidden_n_elem, int num_layers,
      const Dropout<TensorT>& dropout, cudnnRNNInputMode_t input_mode,
      cudnnDirectionMode_t direction, cudnnRNNMode_t mode, cudnnRNNAlgo_t algo,
      cudnnDataType_t dtype) {
    CUDNNXX_DNN_CHECK(cudnnCreateRNNDescriptor(&desc_));
    // NOTE: cuDNN documentation is wrong.
    //       This function takes 10 arguments in fact.
    CUDNNXX_DNN_CHECK(cudnnSetRNNDescriptor_v6(
        handle.raw_handle(), desc_, hidden_n_elem, num_layers, dropout.desc(),
        input_mode, direction, mode, algo, dtype));
  }

  ~RNN() { CUDNNXX_DNN_CHECK(cudnnDestroyRNNDescriptor(desc_)); }

  RNN(const RNN&) = delete;
  RNN operator=(const RNN&) = delete;

  cudnnRNNDescriptor_t desc() const { return desc_; }

  size_t GetParamsSize(const Handle& handle, const TensorArray<TensorT>& x,
                       cudnnDataType_t dtype) const {
    size_t n_bytes = 0;
    CUDNNXX_DNN_CHECK(cudnnGetRNNParamsSize(
        handle.raw_handle(), desc_, x.descs()[0], &n_bytes, dtype));
    return n_bytes;
  }

  size_t GetTrainingReserveSize(const Handle& handle, int seq_length,
                                const TensorArray<TensorT>& x) const {
    size_t n_bytes = 0;
    CUDNNXX_DNN_CHECK(cudnnGetRNNTrainingReserveSize(
        handle.raw_handle(), desc_, seq_length, x.descs(), &n_bytes));
    return n_bytes;
  }

  size_t GetWorkspaceSize(const Handle& handle, int seq_length,
                          const TensorArray<TensorT>& x) const {
    size_t n_bytes = 0;
    CUDNNXX_DNN_CHECK(cudnnGetRNNWorkspaceSize(
        handle.raw_handle(), desc_, seq_length, x.descs(), &n_bytes));
    return n_bytes;
  }

  void ForwardTraining(const Handle& handle, int seq_length,
                       const TensorArray<TensorT>& x, const Tensor<TensorT>& hx,
                       const Tensor<TensorT>& cx, const Filter<TensorT>& w,
                       TensorArray<TensorT>* y, Tensor<TensorT>* hy,
                       Tensor<TensorT>* cy, void* workspace,
                       size_t workspace_n_bytes, void* reserve_space,
                       size_t reserve_space_n_bytes) const {
    CUDNNXX_DNN_CHECK(cudnnRNNForwardTraining(
        handle.raw_handle(), desc_, seq_length, x.descs(), x.dev_mem(),
        hx.desc(), hx.dev_mem(), cx.desc(), cx.dev_mem(), w.desc(), w.dev_mem(),
        y->descs(), y->dev_mem(), hy->desc(), hy->dev_mem(), cy->desc(),
        cy->dev_mem(), workspace, workspace_n_bytes, reserve_space,
        reserve_space_n_bytes));
  }

  void ForwardInference(const Handle& handle, int seq_length,
                        const TensorArray<TensorT>& x,
                        const Tensor<TensorT>& hx, const Tensor<TensorT>& cx,
                        const Filter<TensorT>& w, TensorArray<TensorT>* y,
                        Tensor<TensorT>* hy, Tensor<TensorT>* cy,
                        void* workspace, size_t workspace_n_bytes) const {
    CUDNNXX_DNN_CHECK(cudnnRNNForwardInference(
        handle.raw_handle(), desc_, seq_length, x.descs(), x.dev_mem(),
        hx.desc(), hx.dev_mem(), cx.desc(), cx.dev_mem(), w.desc(), w.dev_mem(),
        y->descs(), y->dev_mem(), hy->desc(), hy->dev_mem(), cy->desc(),
        cy->dev_mem(), workspace, workspace_n_bytes));
  }

  // TODO:
  // cudnnRNNForwardTrainingEx
  // cudnnRNNForwardInferenceEx

  void BackwardData(const Handle& handle, int seq_length,
                    const TensorArray<TensorT>& y,
                    const TensorArray<TensorT>& dy, const Tensor<TensorT>& dhy,
                    const Tensor<TensorT>& dcy, const Filter<TensorT>& w,
                    const Tensor<TensorT>& hx, const Tensor<TensorT>& cx,
                    TensorArray<TensorT>* dx, Tensor<TensorT>* dhx,
                    Tensor<TensorT>* dcx, void* workspace,
                    size_t workspace_n_bytes, void* reserve_space,
                    size_t reserve_space_n_bytes) {
    CUDNNXX_DNN_CHECK(cudnnRNNBackwardData(
        handle.raw_handle(), desc_, seq_length, y.descs(), y.dev_mem(),
        dy.descs(), dy.dev_mem(), dhy.desc(), dhy.dev_mem(), dcy.desc(),
        dcy.dev_mem(), w.desc(), w.dev_mem(), hx.desc(), hx.dev_mem(),
        cx.desc(), cx.dev_mem(), dx->descs(), dx->dev_mem(), dhx->desc(),
        dhx->dev_mem(), dcx->desc(), dcx->dev_mem(), workspace,
        workspace_n_bytes, reserve_space, reserve_space_n_bytes));
  }

  void BackwardWeights(const Handle& handle, int seq_length,
                       const TensorArray<TensorT>& x, const Tensor<TensorT>& hx,
                       const TensorArray<TensorT>& y, void* workspace,
                       size_t workspace_n_bytes,
                       const Filter<TensorT>* dw, void* reserve_space,
                       size_t reserve_space_n_bytes) {
    CUDNNXX_DNN_CHECK(cudnnRNNBackwardWeights(
        handle.raw_handle(), desc_, seq_length, x.descs(), x.dev_mem(),
        hx.desc(), hx.dev_mem(), y.descs(), y.dev_mem(), workspace,
        workspace_n_bytes, dw->desc(), dw->dev_mem(), reserve_space,
        reserve_space_n_bytes));
  }

  // TODO:
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
