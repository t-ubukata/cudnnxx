#include "cudnnxx/dropout.h"
#include "cudnnxx/rnn.h"
#include "gtest/gtest.h"

namespace cudnnxx {

class RNNTest : public ::testing::Test {
 protected:
  Handle handle;
};

TEST_F(RNNTest, TestConstructor) {
  float dropout_p = 0.5;
  unsigned long long seed = 20200627;
  Dropout<float> dropout(handle, dropout_p, seed);
  RNN<float, float> rnn(handle, 1, 1, dropout, CUDNN_LINEAR_INPUT, CUDNN_UNIDIRECTIONAL,
  CUDNN_RNN_RELU, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT);
}

}  // namespace cudnnxx
