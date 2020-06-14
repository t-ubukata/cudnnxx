#include "cudnnxx/rnn.h"
#include "gtest/gtest.h"

namespace cudnnxx {

class RNNTest : public ::testing::Test {
 protected:
  Handle handle;
};

TEST_F(RNNTest, TestConstructor) {
  // RNN<float, float> rnn(handle);
}

}  // namespace cudnnxx
