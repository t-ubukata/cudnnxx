#ifndef CUDNNXX_UTIL_H_
#define CUDNNXX_UTIL_H_

#include <chrono>
#include <ctime>
#include <iostream>

#include "cudnn.h"

#define CUDNNXX_LOG(LEVEL_CHAR, MSG)                                         \
  do {                                                                       \
    const std::chrono::system_clock::time_point tp =                         \
        std::chrono::system_clock::now();                                    \
    const std::time_t tt = std::chrono::system_clock::to_time_t(tp);         \
    char time_str[20];                                                       \
    std::strftime(time_str, sizeof(time_str), "%F %T", std::localtime(&tt)); \
    std::cerr << time_str << ": "                                             \
              << #LEVEL_CHAR << " " << __FILE__ << ":" << __LINE__ << "] "       \
              << (MSG) << "\n";                                              \
  } while (false)

#define CUDNNXX_LOG_INFO(MSG) CUDNNXX_LOG(I, (MSG))
#define CUDNNXX_LOG_WARN(MSG) CUDNNXX_LOG(W, (MSG))
#define CUDNNXX_LOG_ERROR(MSG) CUDNNXX_LOG(E, (MSG))

#define CUDNNXX_LOG_FATAL(MSG) \
  do {                         \
    CUDNNXX_LOG(F, (MSG));     \
    abort();                   \
  } while (false)

#define CUDNNXX_CUDA_CHECK(EXPR)                                      \
  do {                                                                \
    const auto error = (EXPR);                                        \
    if (error != cudaSuccess) {                                       \
      CUDNNXX_LOG_FATAL(std::string(cudaGetErrorName(error)) + ": " + \
                        std::string(cudaGetErrorString(error)));      \
    }                                                                 \
  } while (false)

#define CUDNNXX_DNN_CHECK(EXPR)                     \
  do {                                              \
    const auto stat = (EXPR);                       \
    if (stat != CUDNN_STATUS_SUCCESS) {             \
      CUDNNXX_LOG_FATAL(cudnnGetErrorString(stat)); \
    }                                               \
  } while (false)

#define CUDNNXX_UNUSED_VAR(VAR) static_cast<void>((VAR));

#define CUDNNXX_CHECK(COND, MSG) \
  do {                           \
    if (!(COND)) {               \
      CUDNNXX_LOG_FATAL((MSG));  \
    }                            \
  } while (false)

#endif  // CUDNNXX_UTIL_H_
