#include "cudnnxx/divisive_normalization.h"

#include "gtest/gtest.h"

namespace cudnnxx {

class DivisiveNormalizationTest : public ::testing::Test {
 protected:
  Handle handle;
};

TEST_F(DivisiveNormalizationTest , TestConstructor) {
  DivisiveNormalization<float, float> dn(5, 1e-4, 0.75, 2);
}

}  // namespace cudnnxx
