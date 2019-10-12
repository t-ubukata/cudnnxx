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

  ~Convolution() {
    CUXX_DNN_CHECK(cudnnDestroyConvolutionDescriptor(desc_));
  }

  Convolution(const Convolution&) = delete;
  Convolution operator=(const Convolution&) = delete;

  cudnnConvolutionDescriptor_t desc() {return desc_;}


 private:
  cudnnConvolutionDescriptor_t desc_;
};

}  // namespace dnn
}  // namespace cuxx

#endif  // CUXX_DNN_CONVOLUTION_H_
