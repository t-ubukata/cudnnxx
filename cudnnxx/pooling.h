#ifndef CUDNNXX_POOLING_H_
#define CUDNNXX_POOLING_H_

#include <array>
#include <vector>

#include "cudnn.h"
#include "cudnnxx/common.h"
#include "cudnnxx/util.h"

namespace cudnnxx {

// FactorT must be float or double.
template <typename TensorT, typename FactorT>
class Pooling {
 public:
  // 2d
  Pooling(cudnnPoolingMode_t mode, cudnnNanPropagation_t nan_opt, int window_h,
          int window_w, int vertical_pad, int horizontal_pad,
          int vertical_stride, int horizontal_stride) {
    CUDNNXX_DNN_CHECK(cudnnCreatePoolingDescriptor(&desc_));
    CUDNNXX_DNN_CHECK(cudnnSetPooling2dDescriptor(
        desc_, mode, nan_opt, window_h, window_w, vertical_pad, horizontal_pad,
        vertical_stride, horizontal_stride));
  }

  // Nd
  Pooling(cudnnPoolingMode_t mode, cudnnNanPropagation_t nan_opt, int n_dims,
          const int window_dims[], const int paddings[], const int strides[]) {
    CUDNNXX_DNN_CHECK(cudnnCreatePoolingDescriptor(&desc_));
    // Note: cuDNN API reference is wrong.
    CUDNNXX_DNN_CHECK(cudnnSetPoolingNdDescriptor(
        desc_, mode, nan_opt, n_dims, window_dims, paddings, strides));
  }

  ~Pooling() { CUDNNXX_DNN_CHECK(cudnnDestroyPoolingDescriptor(desc_)); }

  Pooling(const Pooling&) = delete;
  Pooling operator=(const Pooling&) = delete;

  cudnnPoolingDescriptor_t desc() const { return desc_; }

  std::array<int, 4> Get2dForwardOutputDim(const Tensor<TensorT>& in) {
    int n = 0;
    int c = 0;
    int h = 0;
    int w = 0;
    CUDNNXX_DNN_CHECK(
        cudnnGetPooling2dForwardOutputDim(desc_, in.desc(), &n, &c, &h, &w));
    return {{n, c, h, w}};
  }

  std::vector<int> GetNdForwardOutputDim(const Tensor<TensorT>& in,
                                         int n_dims) {
    std::vector<int> out_dims(n_dims);
    CUDNNXX_DNN_CHECK(cudnnGetPoolingNdForwardOutputDim(
        desc_, in.desc(), n_dims, out_dims.data()));
    return out_dims;
  }

  void Forward(const Handle& handle, FactorT alpha, const Tensor<TensorT>& x,
               FactorT beta, Tensor<TensorT>* y) {
    CUDNNXX_DNN_CHECK(cudnnPoolingForward(handle.raw_handle(), desc_, &alpha,
                                          x.desc(), x.dev_mem(), &beta,
                                          y->desc(), y->dev_mem()));
  }

  void Backward(const Handle& handle, FactorT alpha, const Tensor<TensorT>& y,
                const Tensor<TensorT>& dy, const Tensor<TensorT>& x,
                FactorT beta, Tensor<TensorT>* dx) {
    CUDNNXX_DNN_CHECK(cudnnPoolingBackward(
        handle.raw_handle(), desc_, &alpha, y.desc(), y.dev_mem(), dy.desc(),
        dy.dev_mem(), x.desc(), x.dev_mem(), &beta, dx->desc(), dx->dev_mem()));
  }

 private:
  cudnnPoolingDescriptor_t desc_;
};

}  // namespace cudnnxx

#endif  // CUDNNXX_POOLING_H_
