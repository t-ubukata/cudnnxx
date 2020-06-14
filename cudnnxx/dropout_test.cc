#include "cudnnxx/dropout.h"
#include "gtest/gtest.h"

namespace cudnnxx {

class DropoutTest : public ::testing::Test {
 protected:
  Handle handle;
};

TEST_F(DropoutTest, TestConstructor) {
  // Dropout<float, float> dropout(handle);
}

}  // namespace cudnnxx
