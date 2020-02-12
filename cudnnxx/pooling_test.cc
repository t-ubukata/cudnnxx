#include "gtest/gtest.h"
#include "cudnnxx/pooling.h"

namespace cudnnxx {

class PoolingTest : public ::testing::Test {
 protected:
  Handle handle;
};

TEST_F(PoolingTest, TestConstructor2d) {
  Pooling<float, float> pool(CUDNN_POOLING_MAX_DETERMINISTIC,
                             CUDNN_PROPAGATE_NAN, 2, 2, 1, 1, 2, 2);
}

TEST_F(PoolingTest, TestConstructorNd) {
  constexpr int n_dims = 3;
  int window_dims[n_dims] = {2, 2, 2};
  int paddings[n_dims] = {1, 1, 1};
  int strides[n_dims] = {2, 2, 2};
  Pooling<float, float> pool(CUDNN_POOLING_MAX_DETERMINISTIC,
                             CUDNN_PROPAGATE_NAN, n_dims, window_dims,
                             paddings, strides);
}

}  // namespace cudnnxx
