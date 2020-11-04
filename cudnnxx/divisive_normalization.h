#ifndef CUDNNXX_DIVISIVE_NORMALIZATION_H_
#define CUDNNXX_DIVISIVE_NORMALIZATION_H_

#include "cudnn.h"
#include "cudnnxx/common.h"
#include "cudnnxx/util.h"

namespace cudnnxx {

// FactorT must be float or double.
template <typename TensorT, typename FactorT>
class DivisiveNormalization {
 public:
  DivisiveNormalization (unsigned int n, double alpha, double beta, double k) {
    CUDNNXX_DNN_CHECK(cudnnCreateLRNDescriptor(&desc_));
    CUDNNXX_DNN_CHECK(cudnnSetLRNDescriptor(desc_, n, alpha, beta, k));
  }

  ~DivisiveNormalization () { CUDNNXX_DNN_CHECK(cudnnDestroyLRNDescriptor(desc_)); }

  DivisiveNormalization (const DivisiveNormalization &) = delete;
  DivisiveNormalization  operator=(const DivisiveNormalization &) = delete;

  cudnnLRNDescriptor_t desc() const { return desc_; }



 private:
  cudnnLRNDescriptor_t desc_;
};

}  // namespace cudnnxx

#endif  // CUDNNXX_DIVISIVE_NORMALIZATION_H_
