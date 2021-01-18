#include "cudnnxx/ctc_loss.h"

#include "gtest/gtest.h"

namespace cudnnxx {

class CTCLossTest : public ::testing::Test {
 protected:
  Handle handle;
};

TEST_F(CTCLossTest, TestConstructor) {
  CTCLoss<float, float> ctcl(CUDNN_DATA_FLOAT);
}

TEST_F(CTCLossTest, TestConstructor2) {
  CTCLoss<float, float> ctcl(CUDNN_DATA_FLOAT,CUDNN_LOSS_NORMALIZATION_NONE,
                             CUDNN_NOT_PROPAGATE_NAN);
}

}  // namespace cudnnxx
