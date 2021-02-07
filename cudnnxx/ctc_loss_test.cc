#include "cudnnxx/ctc_loss.h"

#include <vector>

#include "gtest/gtest.h"

namespace cudnnxx {

class CTCLossTest : public ::testing::Test {
 protected:
  Handle handle;
};

TEST_F(CTCLossTest, TestConstructor) { CTCLoss<float> ctcl(CUDNN_DATA_FLOAT); }

TEST_F(CTCLossTest, TestConstructor2) {
  CTCLoss<float> ctcl(CUDNN_DATA_FLOAT, CUDNN_LOSS_NORMALIZATION_NONE,
                      CUDNN_NOT_PROPAGATE_NAN);
}

// TODO: Value check.
TEST_F(CTCLossTest, TestGetWorkspaceSize) {
  CTCLoss<float> ctcl(CUDNN_DATA_FLOAT, CUDNN_LOSS_NORMALIZATION_NONE,
                      CUDNN_NOT_PROPAGATE_NAN);
  constexpr int t = 50;  // input length
  constexpr int c = 20;  // number of classes
  constexpr int n = 16;  // batch size
  // constexpr int s = 30;  // max target length

  constexpr int probs_n_elem = t * n * c;
  size_t probs_size_in_bytes = sizeof(float) * probs_n_elem;

  float probs_mem_host[probs_n_elem] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                        0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                                        1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4};
  float* probs_mem_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&probs_mem_dev, probs_size_in_bytes));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(probs_mem_dev, probs_mem_host,
                                probs_size_in_bytes, cudaMemcpyHostToDevice));
  constexpr int probs_n_dims = 3;
  int probs_dims[probs_n_dims] = {t, n, c};
  int probs_strides[probs_n_dims] = {probs_dims[1] * probs_dims[2],
                                     probs_dims[2], 1};
  Tensor<float> probs(CUDNN_DATA_FLOAT, probs_n_dims, probs_dims, probs_strides,
                      probs_mem_dev);

  constexpr int gradients_n_elem = probs_n_elem;
  size_t gradients_size_in_bytes = sizeof(float) * gradients_n_elem;

  float gradients_mem_host[gradients_n_elem] = {0};
  float* gradients_mem_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&gradients_mem_dev, gradients_size_in_bytes));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(gradients_mem_dev, gradients_mem_host,
                                gradients_size_in_bytes,
                                cudaMemcpyHostToDevice));
  constexpr int gradients_n_dims = 3;
  int gradients_dims[gradients_n_dims] = {t, n, c};
  int gradients_strides[gradients_n_dims] = {
      gradients_dims[1] * gradients_dims[2], gradients_dims[2], 1};
  Tensor<float> gradients(CUDNN_DATA_FLOAT, gradients_n_dims, gradients_dims,
                          gradients_strides, gradients_mem_dev);

  int labels_lengths[n] = {};
  for (int i = 0; i < n; ++i) {
    labels_lengths[i] = i;
  }

  int labels_n_elem = 0;
  for (int i = 0; i < n; ++i) {
    labels_n_elem += labels_lengths[i];
  }
  std::vector<int> labels(labels_n_elem);
  for (int i = 0; i < labels_n_elem; ++i) {
    labels[i] = i;
  }

  int input_lengths[n] = {t};

  ctcl.GetWorkspaceSize(handle, probs, gradients, labels.data(), labels_lengths,
                        input_lengths, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC);
}

// TODO: Value check.
TEST_F(CTCLossTest, TestCompute) {
  CTCLoss<float> ctcl(CUDNN_DATA_FLOAT, CUDNN_LOSS_NORMALIZATION_NONE,
                      CUDNN_NOT_PROPAGATE_NAN);
  constexpr int t = 50;  // input length
  constexpr int c = 20;  // number of classes
  constexpr int n = 16;  // batch size
  // constexpr int s = 30;  // max target length

  constexpr int probs_n_elem = t * n * c;
  size_t probs_size_in_bytes = sizeof(float) * probs_n_elem;

  float probs_mem_host[probs_n_elem] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                        0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                                        1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4};
  float* probs_mem_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&probs_mem_dev, probs_size_in_bytes));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(probs_mem_dev, probs_mem_host,
                                probs_size_in_bytes, cudaMemcpyHostToDevice));
  constexpr int probs_n_dims = 3;
  int probs_dims[probs_n_dims] = {t, n, c};
  int probs_strides[probs_n_dims] = {probs_dims[1] * probs_dims[2],
                                     probs_dims[2], 1};
  Tensor<float> probs(CUDNN_DATA_FLOAT, probs_n_dims, probs_dims, probs_strides,
                      probs_mem_dev);

  constexpr int gradients_n_elem = probs_n_elem;
  size_t gradients_size_in_bytes = sizeof(float) * gradients_n_elem;

  float gradients_mem_host[gradients_n_elem] = {0};
  float* gradients_mem_dev = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&gradients_mem_dev, gradients_size_in_bytes));
  CUDNNXX_CUDA_CHECK(cudaMemcpy(gradients_mem_dev, gradients_mem_host,
                                gradients_size_in_bytes,
                                cudaMemcpyHostToDevice));
  constexpr int gradients_n_dims = 3;
  int gradients_dims[gradients_n_dims] = {t, n, c};
  int gradients_strides[gradients_n_dims] = {
      gradients_dims[1] * gradients_dims[2], gradients_dims[2], 1};
  Tensor<float> gradients(CUDNN_DATA_FLOAT, gradients_n_dims, gradients_dims,
                          gradients_strides, gradients_mem_dev);

  int labels_lengths[n] = {};
  for (int i = 0; i < n; ++i) {
    labels_lengths[i] = i;
  }

  int labels_n_elem = 0;
  for (int i = 0; i < n; ++i) {
    labels_n_elem += labels_lengths[i];
  }
  std::vector<int> labels(labels_n_elem);
  for (int i = 0; i < labels_n_elem; ++i) {
    labels[i] = i;
  }

  int input_lengths[n] = {t};
  auto algo = CUDNN_CTC_LOSS_ALGO_DETERMINISTIC;
  auto workspace_size_in_bytes =
      ctcl.GetWorkspaceSize(handle, probs, gradients, labels.data(),
                            labels_lengths, input_lengths, algo);

  void* workspace;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&workspace, workspace_size_in_bytes));
  void* costs = nullptr;
  CUDNNXX_CUDA_CHECK(cudaMalloc(&costs, n));
  ctcl.Compute(handle, probs, labels.data(), labels_lengths, input_lengths,
               costs, &gradients, algo, workspace, workspace_size_in_bytes);

  CUDNNXX_CUDA_CHECK(cudaFree(costs));
  CUDNNXX_CUDA_CHECK(cudaFree(workspace));
  CUDNNXX_CUDA_CHECK(cudaFree(probs_mem_dev));
}

}  // namespace cudnnxx
