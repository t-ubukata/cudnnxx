CXX := clang++
CUDA_DIR := /usr/local/cuda
INCLUDE_DIR := $(CUDA_DIR)/include
LIB_DIR := $(CUDA_DIR)/lib64
# LIBLARIES := cudart cudnn
CXXFLAGS := -g -std=c++11 -pedantic -Wall -Wextra -fno-exceptions -fPIC -I $(INCLUDE_DIR)
TARGET_LIB := ./lib/libcuxx.so
BUILD_DIR := ./build
OBJS := $(BUILD_DIR)/dnn.o
LDFLAGS := -shared

$(TARGET_LIB): $(OBJS)
	$(LD) $(LDFLAGS) $^ -o $(TARGET_LIB)

SRC_DIR := ./src/cuxx

$(BUILD_DIR)/dnn.o: $(SRC_DIR)/dnn/common.cc
	$(CXX) $(CXXFLAGS) $^ -c -o $(BUILD_DIR)/dnn.o

.PHONY: clean
clean:
	$(RM) $(BUILD_DIR)/* $(TARGET_LIB)
