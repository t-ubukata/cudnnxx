#ifndef CUDNNXX_CONVOLUTION_H_
#define CUDNNXX_CONVOLUTION_H_

#include "cudnn.h"
#include "cudnnxx/common.h"
#include "cudnnxx/util.h"

namespace cudnnxx {

template <typename TensorT, typename FactorT>
class Convolution {
 public:
  // 2d
  Convolution(int pad_h, int pad_w, int u, int v, int dilation_h,
              int dilation_w, cudnnConvolutionMode_t mode,
              cudnnDataType_t dtype) {
    CUDNNXX_DNN_CHECK(cudnnCreateConvolutionDescriptor(&desc_));
    CUDNNXX_DNN_CHECK(cudnnSetConvolution2dDescriptor(
        desc_, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, dtype));
  }

  // Nd
  Convolution(int array_length, const int pads[], const int filter_strides[],
              const int dilations[], cudnnConvolutionMode_t mode,
              cudnnDataType_t dtype) {
    CUDNNXX_DNN_CHECK(cudnnCreateConvolutionDescriptor(&desc_));
    CUDNNXX_DNN_CHECK(cudnnSetConvolutionNdDescriptor(
        desc_, array_length, pads, filter_strides, dilations, mode, dtype));
  }

  ~Convolution() {
    CUDNNXX_DNN_CHECK(cudnnDestroyConvolutionDescriptor(desc_));
  }

  Convolution(const Convolution&) = delete;
  Convolution operator=(const Convolution&) = delete;

  cudnnConvolutionDescriptor_t desc() const { return desc_; }

  void SetGroupCount(int group_count) {
    CUDNNXX_DNN_CHECK(cudnnSetConvolutionGroupCount(desc_, group_count));
  }

  int GetGroupCount() {
    int count = 0;
    CUDNNXX_DNN_CHECK(cudnnGetConvolutionGroupCount(desc_, &count));
    return count;
  }

  void SetMathType(cudnnMathType_t math_type) {
    CUDNNXX_DNN_CHECK(cudnnSetConvolutionMathType(desc_, math_type));
  }

  cudnnMathType_t GetMathType() const {
    cudnnMathType_t type;
    CUDNNXX_DNN_CHECK(cudnnGetConvolutionMathType(desc_, &type));
    return type;
  }

  static int GetForwardAlgorithmMaxCount(const Handle& handle) {
    int count = 0;
    CUDNNXX_DNN_CHECK(cudnnGetConvolutionForwardAlgorithmMaxCount(
        handle.raw_handle(), &count));
    return count;
  }

  void GetForwardAlgorithm(const Handle& handle, const Tensor<TensorT>& x,
                           const Filter<TensorT>& w, const Tensor<TensorT>& y,
                           const int requested_algo_count,
                           int* returned_algo_count,
                           cudnnConvolutionFwdAlgoPerf_t* results) const {
    CUDNNXX_DNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
        handle.raw_handle(), x.desc(), w.desc(), desc_, y.desc(),
        requested_algo_count, returned_algo_count, results));
  }

  size_t GetForwardWorkspaceSize(const Handle& handle, const Tensor<TensorT>& x,
                                 const Filter<TensorT>& w,
                                 const Tensor<TensorT>& y,
                                 cudnnConvolutionFwdAlgo_t algo) const {
    size_t n_bytes = 0;
    CUDNNXX_DNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        handle.raw_handle(), x.desc(), w.desc(), desc_, y.desc(), algo,
        &n_bytes));
    return n_bytes;
  }

  void FindForwardAlgorithm(const Handle& handle, const Tensor<TensorT>& x,
                            const Filter<TensorT>& w, const Tensor<TensorT>& y,
                            const int requested_algo_count,
                            int* returned_algo_count,
                            cudnnConvolutionFwdAlgoPerf_t* results,
                            void* workspace, size_t workspace_n_bytes) const {
    CUDNNXX_DNN_CHECK(cudnnFindConvolutionForwardAlgorithmEx(
        handle.raw_handle(), x.desc(), x.dev_mem(), w.desc(), w.dev_mem(),
        desc_, y.desc(), y.dev_mem(), requested_algo_count, returned_algo_count,
        results, workspace, workspace_n_bytes));
  }

  void Get2dForwardOutputDim(const Tensor<TensorT>& input,
                             const Filter<TensorT>& filter, int* n, int* c,
                             int* h, int* w) const {
    CUDNNXX_DNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
        desc_, input.desc(), filter.desc(), n, c, h, w));
  }

  void GetNdForwardOutputDim(const Tensor<TensorT>& input,
                             const Filter<TensorT>& filter, int n_dims,
                             int output_dims[]) const {
    CUDNNXX_DNN_CHECK(cudnnGetConvolutionNdForwardOutputDim(
        desc_, input.desc(), filter.desc(), n_dims, output_dims));
  }

  void Forward(const Handle& handle, FactorT alpha, const Tensor<TensorT>& x,
               const Filter<TensorT>& w, cudnnConvolutionFwdAlgo_t algo,
               void* workspace, size_t workspace_n_bytes, FactorT beta,
               Tensor<TensorT>* y) const {
    CUDNNXX_DNN_CHECK(cudnnConvolutionForward(
        handle.raw_handle(), &alpha, x.desc(), x.dev_mem(), w.desc(),
        w.dev_mem(), desc_, algo, workspace, workspace_n_bytes, &beta,
        y->desc(), y->dev_mem()));
  }

  // TODO: cudnnConvolutionBiasActivationForward

  static int GetBackwardDataAlgorithmMaxCount(const Handle& handle) {
    int count = 0;
    CUDNNXX_DNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
        handle.raw_handle(), &count));
    return count;
  }

  void GetBackwardDataAlgorithm(
      const Handle& handle, const Filter<TensorT>& w, const Tensor<TensorT>& dy,
      const Tensor<TensorT>& dx, const int requested_algo_count,
      int* returned_algo_count,
      cudnnConvolutionBwdDataAlgoPerf_t* results) const {
    CUDNNXX_DNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        handle.raw_handle(), w.desc(), dy.desc(), desc_, dx.desc(),
        requested_algo_count, returned_algo_count, results));
  }

  size_t GetBackwardDataWorkspaceSize(
      const Handle& handle, const Filter<TensorT>& w, const Tensor<TensorT>& dy,
      const Tensor<TensorT>& dx, cudnnConvolutionBwdDataAlgo_t algo) const {
    size_t n_bytes = 0;
    CUDNNXX_DNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle.raw_handle(), w.desc(), dy.desc(), desc_, dx.desc(), algo,
        &n_bytes));
    return n_bytes;
  }

  void FindBackwardDataAlgorithm(
      const Handle& handle, const Filter<TensorT>& w, const Tensor<TensorT>& dy,
      const Tensor<TensorT>& dx, const int requested_algo_count,
      int* returned_algo_count, cudnnConvolutionBwdDataAlgoPerf_t* results,
      void* workspace, size_t workspace_n_bytes) const {
    CUDNNXX_DNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithmEx(
        handle.raw_handle(), w.desc(), w.dev_mem(), dy.desc(), dy.dev_mem(),
        desc_, dx.desc(), dx.dev_mem(), requested_algo_count,
        returned_algo_count, results, workspace, workspace_n_bytes));
  }

  void BackwardData(const Handle& handle, FactorT alpha,
                    const Filter<TensorT>& w, const Tensor<TensorT>& dy,
                    cudnnConvolutionBwdDataAlgo_t algo, void* workspace,
                    size_t workspace_n_bytes, FactorT beta,
                    Tensor<TensorT>* dx) const {
    CUDNNXX_DNN_CHECK(cudnnConvolutionBackwardData(
        handle.raw_handle(), &alpha, w.desc(), w.dev_mem(), dy.desc(),
        dy.dev_mem(), desc_, algo, workspace, workspace_n_bytes, &beta,
        dx->desc(), dx->dev_mem()));
  }

  static int GetBackwardFilterAlgorithmMaxCount(const Handle& handle) {
    int count = 0;
    CUDNNXX_DNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
        handle.raw_handle(), &count));
    return count;
  }

  void GetBackwardFilterAlgorithm(
      const Handle& handle, const Tensor<TensorT>& x, const Tensor<TensorT>& dy,
      const Filter<TensorT>& dw, const int requested_algo_count,
      int* returned_algo_count,
      cudnnConvolutionBwdFilterAlgoPerf_t* results) const {
    CUDNNXX_DNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        handle.raw_handle(), x.desc(), dy.desc(), desc_, dw.desc(),
        requested_algo_count, returned_algo_count, results));
  }

  size_t GetBackwardFilterWorkspaceSize(
      const Handle& handle, const Tensor<TensorT>& x, const Tensor<TensorT>& dy,
      const Filter<TensorT>& dw, cudnnConvolutionBwdFilterAlgo_t algo) const {
    size_t n_bytes = 0;
    CUDNNXX_DNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle.raw_handle(), x.desc(), dy.desc(), desc_, dw.desc(), algo,
        &n_bytes));
    return n_bytes;
  }

  void FindBackwardFilterAlgorithm(
      const Handle& handle, const Tensor<TensorT>& x, const Tensor<TensorT>& dy,
      const Filter<TensorT>& dw, const int requested_algo_count,
      int* returned_algo_count, cudnnConvolutionBwdFilterAlgoPerf_t* results,
      void* workspace, size_t workspace_n_bytes) const {
    CUDNNXX_DNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithmEx(
        handle.raw_handle(), x.desc(), x.dev_mem(), dy.desc(), dy.dev_mem(),
        desc_, dw.desc(), dw.dev_mem(), requested_algo_count,
        returned_algo_count, results, workspace, workspace_n_bytes));
  }

  void BackwardFilter(const Handle& handle, FactorT alpha,
                      const Tensor<TensorT>& x, const Tensor<TensorT>& dy,
                      cudnnConvolutionBwdFilterAlgo_t algo, void* workspace,
                      size_t workspace_n_bytes, FactorT beta,
                      Filter<TensorT>* dw) const {
    CUDNNXX_DNN_CHECK(cudnnConvolutionBackwardFilter(
        handle.raw_handle(), &alpha, x.desc(), x.dev_mem(), dy.desc(),
        dy.dev_mem(), desc_, algo, workspace, workspace_n_bytes, &beta,
        dw->desc(), dw->dev_mem()));
  }

  // cudnnConvolutionBackwardBias

 private:
  cudnnConvolutionDescriptor_t desc_;
};

}  // namespace cudnnxx

#endif  // CUDNNXX_CONVOLUTION_H_
