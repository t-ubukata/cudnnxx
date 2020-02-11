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
  // Pooling(int array_length, const int pads[], const int filter_strides[],
  //             const int dilations[],
  //             cudnnConvolutionMode_t mode, cudnnDataType_t dtype) {
  //   CUDNNXX_DNN_CHECK(cudnnCreateConvolutionDescriptor(&desc_));
  //   CUDNNXX_DNN_CHECK(cudnnSetConvolutionNdDescriptor(desc_, array_length, pads,
  //                                                  filter_strides, dilations,
  //                                                  mode, dtype));
  // }

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
