#ifndef CUDNNXX_SPATIAL_TRANSFORMER_H_
#define CUDNNXX_SPATIAL_TRANSFORMER_H_

#include "cudnn.h"
#include "cudnnxx/common.h"
#include "cudnnxx/util.h"

namespace cudnnxx {

template <typename TensorT, typename FactorT, typename ThetaT, typename GridT>
class SpatialTransformer {
 public:
  SpatialTransformer(cudnnDataType_t dtype, int n_dims, int dims[]) {
    CUDNNXX_DNN_CHECK(cudnnCreateSpatialTransformerDescriptor(&desc_));
    CUDNNXX_DNN_CHECK(cudnnSetSpatialTransformerNdDescriptor(
        desc_, CUDNN_SAMPLER_BILINEAR, dtype, n_dims, dims));
  }

  ~SpatialTransformer() {
    CUDNNXX_DNN_CHECK(cudnnDestroySpatialTransformerDescriptor(desc_));
  }

  SpatialTransformer(const SpatialTransformer&) = delete;
  SpatialTransformer operator=(const SpatialTransformer&) = delete;

  cudnnSpatialTransformerDescriptor_t desc() const { return desc_; }

  void GridGeneratorForward(const Handle& handle, ThetaT* theta, GridT* grid) {
    CUDNNXX_DNN_CHECK(cudnnSpatialTfGridGeneratorForward(handle.raw_handle(),
                                                         desc_, theta, grid));
  }

  void GridGeneratorBackward(const Handle& handle, GridT* dgrid,
                             ThetaT* dtheta) {
    CUDNNXX_DNN_CHECK(cudnnSpatialTfGridGeneratorBackward(
        handle.raw_handle(), desc_, dgrid, dtheta));
  }

  void SamplerForward(const Handle& handle, FactorT alpha,
                      const Tensor<TensorT>& x, GridT* grid, FactorT beta,
                      Tensor<TensorT>* y) const {
    CUDNNXX_DNN_CHECK(cudnnSpatialTfSamplerForward(
        handle.raw_handle(), desc_, &alpha, x.desc(), x.dev_mem(), grid, &beta,
        y->desc(), y->dev_mem()));
  }

  void SamplerBackward(const Handle& handle, FactorT alpha,
                       const Tensor<TensorT>& x, FactorT beta,
                       Tensor<TensorT>* dx, FactorT alpha_dgrid,
                       const Tensor<TensorT>& dy, GridT* grid,
                       FactorT beta_dgrid, GridT* dgrid) const {
    CUDNNXX_DNN_CHECK(cudnnSpatialTfSamplerBackward(
        handle.raw_handle(), desc_, &alpha, x.desc(), x.dev_mem(), &beta,
        dx->desc(), dx->dev_mem(), &alpha_dgrid, dy.desc(), dy.dev_mem(), grid,
        &beta_dgrid, dgrid));
  }

 private:
  cudnnSpatialTransformerDescriptor_t desc_;
};

}  // namespace cudnnxx

#endif  // CUDNNXX_SPATIAL_TRANSFORMER_H_
