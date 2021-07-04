#include "cudnnxx/spatial_transformer.h"

#include "gtest/gtest.h"

namespace cudnnxx {

class SpatialTransformerTest : public ::testing::Test {
 protected:
  Handle handle;
};

TEST_F(SpatialTransformerTest, TestConstructor) {
  constexpr int n_dims = 4;
  int dims[n_dims] = {2, 3, 2, 2};
  SpatialTransformer<float, float> st(CUDNN_DATA_FLOAT, n_dims, dims);
}

}  // namespace cudnnxx
