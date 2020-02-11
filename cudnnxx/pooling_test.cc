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

}  // namespace cudnnxx
