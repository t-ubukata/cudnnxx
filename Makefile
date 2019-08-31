CXX := clang++
CUDA_DIR := /usr/local/cuda
# LIB_DIR := $(CUDA_DIR)/lib64
# LIBLARIES := cudart cudnn
CXXFLAGS := -g -std=c++11 -pedantic -Wall -Wextra -fno-exceptions -fPIC -I . \
            -I $(CUDA_DIR)/include
TARGET_LIB := ./lib/libcuxx.so
BUILD_DIR := ./build
OBJS := $(BUILD_DIR)/dnn.o
LDFLAGS := -shared

$(TARGET_LIB): $(OBJS)
	$(CXX) $(LDFLAGS) $^ -o $(TARGET_LIB)

SRC_DIR := ./cuxx

$(BUILD_DIR)/dnn.o: $(SRC_DIR)/dnn/common.cc
	$(CXX) $(CXXFLAGS) $^ -c -o $(BUILD_DIR)/dnn.o

./test_bin/test_main: $(SRC_DIR)/test_main.cc $(SRC_DIR)/dnn/common_test.cc

.PHONY: clean
clean:
	$(RM) $(BUILD_DIR)/* $(TARGET_LIB)
