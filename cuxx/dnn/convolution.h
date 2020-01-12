#ifndef CUXX_DNN_CONVOLUTION_H_
#define CUXX_DNN_CONVOLUTION_H_

#include "cudnn.h"
#include "cuxx/util.h"

#include "cuxx/dnn/common.h"

namespace cuxx {
namespace dnn {

// FactorT must be float or double.
template<typename TensorT, typename FactorT>
class Convolution {
 public:
  // 2d
  Convolution(int pad_h, int pad_w, int u, int v,
              int dilation_h, int dilation_w,
              cudnnConvolutionMode_t mode, cudnnDataType_t dtype) {
    CUXX_DNN_CHECK(cudnnCreateConvolutionDescriptor(&desc_));
    CUXX_DNN_CHECK(cudnnSetConvolution2dDescriptor(desc_, pad_h, pad_w, u, v,
                                                   dilation_h, dilation_w,
                                                   mode, dtype));
  }

  // Nd
  Convolution(int array_length, const int pads[], const int filter_strides[],
              const int dilations[],
              cudnnConvolutionMode_t mode, cudnnDataType_t dtype) {
    CUXX_DNN_CHECK(cudnnCreateConvolutionDescriptor(&desc_));
    CUXX_DNN_CHECK(cudnnSetConvolutionNdDescriptor(desc_, array_length, pads,
                                                   filter_strides, dilations,
                                                   mode, dtype));
  }

  void SetGroupCount(int group_count) {
    CUXX_DNN_CHECK(cudnnSetConvolutionGroupCount(desc_, group_count));
  }

  int GetGroupCount() {
    int count = 0;
    CUXX_DNN_CHECK(cudnnGetConvolutionGroupCount(desc_, &count));
    return count;
  }

  void SetMathType(cudnnMathType_t math_type) {
    CUXX_DNN_CHECK(cudnnSetConvolutionMathType(desc_, math_type));
  }

  cudnnMathType_t GetMathType() const {
    cudnnMathType_t type;
    CUXX_DNN_CHECK(cudnnGetConvolutionMathType(desc_, &type));
    return type;
  }

  static int GetForwardAlgorithmMaxCount(const Handle& handle) {
    int count = 0;
    CUXX_DNN_CHECK(cudnnGetConvolutionForwardAlgorithmMaxCount(
        handle.raw_handle(), &count));
    return count;
  }

  void GetForwardAlgorithm(const Handle& handle,
                           const Tensor<TensorT>& x, const Filter<TensorT>& w,
                           const Tensor<TensorT>& y,
                           const int requested_algo_count,
                           int *returned_algo_count,
                           cudnnConvolutionFwdAlgoPerf_t *results) const {
    CUXX_DNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(handle.raw_handle(),
                                                          x.desc(), w.desc(),
                                                          desc_, y.desc(),
                                                          requested_algo_count,
                                                          returned_algo_count,
                                                          results));
  }

  size_t GetForwardWorkspaceSize(const Handle& handle, const Tensor<TensorT>& x,
                                 const Filter<TensorT>& w,
                                 const Tensor<TensorT>& y,
                                 cudnnConvolutionFwdAlgo_t algo) const {
    size_t size = 0;
    CUXX_DNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle.raw_handle(),
                                                           x.desc(), w.desc(),
                                                           desc_, y.desc(),
                                                           algo, &size));
    return size;
  }

  void FindForwardAlgorithm(const Handle& handle, const Tensor<TensorT>& x,
                            const Filter<TensorT>& w, const Tensor<TensorT>& y,
                            const int requested_algo_count,
                            int* returned_algo_count,
                            cudnnConvolutionFwdAlgoPerf_t* results,
                            void* workspace,
                            size_t workspace_size_in_bytes) const {
    CUXX_DNN_CHECK(cudnnFindConvolutionForwardAlgorithmEx(handle.raw_handle(),
                                                          x.desc(), x.dev_mem(),
                                                          w.desc(), w.dev_mem(),
                                                          desc_,
                                                          y.desc(), y.dev_mem(),
                                                          requested_algo_count,
                                                          returned_algo_count,
                                                          results, workspace,
                                                          workspace_size_in_bytes));
  }

  void Get2dForwardOutputDim(const Tensor<TensorT>& input,
                             const Filter<TensorT>& filter,
                             int* n, int* c, int* h, int* w) const {
    CUXX_DNN_CHECK(cudnnGetConvolution2dForwardOutputDim(desc_, input.desc(),
                                                         filter.desc(),
                                                         n, c, h, w));
  }

  void GetNdForwardOutputDim(const Tensor<TensorT>& input,
                             const Filter<TensorT>& filter,
                             int n_dims, int output_dims[]) const {
    CUXX_DNN_CHECK(cudnnGetConvolutionNdForwardOutputDim(desc_, input.desc(),
                                                         filter.desc(), n_dims,
                                                         output_dims));
  }

  void Forward(const Handle& handle, FactorT alpha, const Tensor<TensorT>& x,
               const Filter<TensorT>& w, cudnnConvolutionFwdAlgo_t algo,
               void* workspace, size_t workspace_size,
               FactorT beta, Tensor<TensorT>* y) const {
    CUXX_DNN_CHECK(cudnnConvolutionForward(handle.raw_handle(),
                                           &alpha, x.desc(), x.dev_mem(),
                                           w.desc(), w.dev_mem(),
                                           desc_, algo,
                                           workspace, workspace_size,
                                           &beta, y->desc(), y->dev_mem()));
  }

  // cudnnConvolutionBiasActivationForward

  // cudnnConvolutionBackwardBias
  // cudnnConvolutionBackwardData
  // cudnnConvolutionBackwardFilter
  // cudnnFindConvolutionBackwardDataAlgorithmEx
  // cudnnFindConvolutionBackwardFilterAlgorithmEx
  // cudnnGetConvolutionBackwardDataAlgorithmMaxCount
  // cudnnGetConvolutionBackwardDataAlgorithm_v7
  // cudnnGetConvolutionBackwardDataWorkspaceSize
  // cudnnGetConvolutionBackwardFilterAlgorithmMaxCount
  // cudnnGetConvolutionBackwardFilterAlgorithm_v7
  // cudnnGetConvolutionBackwardFilterWorkspaceSize

  ~Convolution() {
    CUXX_DNN_CHECK(cudnnDestroyConvolutionDescriptor(desc_));
  }

  Convolution(const Convolution&) = delete;
  Convolution operator=(const Convolution&) = delete;

  cudnnConvolutionDescriptor_t desc() const {return desc_;}

 private:
  cudnnConvolutionDescriptor_t desc_;
};

}  // namespace dnn
}  // namespace cuxx

#endif  // CUXX_DNN_CONVOLUTION_H_
