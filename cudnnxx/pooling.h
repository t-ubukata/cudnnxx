#ifndef CUDNNXX_POOLING_H_
#define CUDNNXX_POOLING_H_

#include "cudnn.h"
#include "cudnnxx/common.h"
#include "cudnnxx/util.h"

namespace cudnnxx {

// FactorT must be float or double.
template<typename TensorT, typename FactorT>
class Pooling {
 public:
  // 2d
  Pooling(cudnnPoolingMode_t mode, cudnnNanPropagation_t nan_opt,
          int window_h, int window_w, int vertical_pad, int horizontal_pad,
          int vertical_stride, int horizontal_stride) {
    CUDNNXX_DNN_CHECK(cudnnCreatePoolingDescriptor(&desc_));
    CUDNNXX_DNN_CHECK(cudnnSetPooling2dDescriptor(desc_, mode, nan_opt,
                                                  window_h, window_w,
                                                  vertical_pad, horizontal_pad,
                                                  vertical_stride,
                                                  horizontal_stride));
  }

  // Nd
  Pooling(cudnnPoolingMode_t mode, cudnnNanPropagation_t nan_opt, int n_dims,
          const int window_dims[], const int paddings[], const int strides[]) {
    CUDNNXX_DNN_CHECK(cudnnCreatePoolingDescriptor(&desc_));
    // Note: API reference is wrong.
    CUDNNXX_DNN_CHECK(cudnnSetPoolingNdDescriptor(desc_, mode, nan_opt, n_dims,
                                                  window_dims, paddings,
                                                  strides));
  }

  ~Pooling() {
    CUDNNXX_DNN_CHECK(cudnnDestroyPoolingDescriptor(desc_));
  }

  Pooling(const Pooling&) = delete;
  Pooling operator=(const Pooling&) = delete;

  cudnnPoolingDescriptor_t desc() const {return desc_;}

  // Get2dForwardOutputDim
  // GetNdForwardOutputDim
  // Forward
  // Backward

 private:
  cudnnPoolingDescriptor_t desc_;
};

}  // namespace cudnnxx

#endif  // CUDNNXX_POOLING_H_
