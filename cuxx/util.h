#ifndef CUXX_UTIL_H_
#define CUXX_UTIL_H_

#include <ctime>
#include <chrono>
#include <iostream>

#include "cudnn.h"

#define CUXX_LOG_FATAL(MSG) \
  do { \
    const std::chrono::system_clock::time_point tp = \
        std::chrono::system_clock::now(); \
    const std::time_t tt = std::chrono::system_clock::to_time_t(tp); \
    char time_str[20]; \
    std::strftime(time_str, sizeof(time_str), "%F %T", std::localtime(&tt)); \
    std::cerr << time_str << ":" << " F " <<  __FILE__ << ":" << \
    __LINE__ << "] " << (MSG) << std::endl; \
    abort(); \
  } while (false)

#define CUXX_CUDA_CHECK(EXPR) \
  do { \
    const auto error = (EXPR); \
    if (error != cudaSuccess) { \
      CUXX_LOG_FATAL(std::string(cudaGetErrorName(error)) + ": " \
                     + std::string(cudaGetErrorString(error))); \
    } \
  } while (false)

#define CUXX_DNN_CHECK(EXPR) \
  do { \
    const auto stat = (EXPR); \
    if (stat != CUDNN_STATUS_SUCCESS) { \
      CUXX_LOG_FATAL(cudnnGetErrorString(stat)); \
    } \
  } while (false)

#define CUXX_UNUSED_VAR(VAR) static_cast<void>((VAR));

#endif  // CUXX_UTIL_H_
