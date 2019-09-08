CXX := clang++
CUDA_DIR := /usr/local/cuda
# LIB_DIR := $(CUDA_DIR)/lib64
# LIBLARIES := cudart cudnn
CXXFLAGS := -g -std=c++11 -pedantic -Wall -Wextra -fno-exceptions -fPIC -I . \
            -I $(CUDA_DIR)/include
TARGET_LIB := ./lib/libcuxx.so
# BUILD_DIR := ./build
OBJ_DIR := ./obj
OBJS := $(OBJ_DIR)/common.o $(OBJ_DIR)/test_main.o $(OBJ_DIR)/common_test.o
LDFLAGS := -L /usr/local/cuda/lib64 -l cuda -l cudart -l cudnn

# The target lib.

$(TARGET_LIB): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -shared $^ -o $@

SRC_DIR := ./cuxx

# Objs.

$(OBJ_DIR)/common.o: $(SRC_DIR)/dnn/common.cc
	$(CXX) $(CXXFLAGS) $^ -c -o $@

# The test main.

./test_bin/test_main: $(OBJ_DIR)/test_main.o $(TARGET_LIB)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

# Test objs.

$(OBJ_DIR)/test_main.o: $(SRC_DIR)/test_main.cc $(TARGET_LIB)
	$(CXX) $(CXXFLAGS) $^ -c -o $@

$(OBJ_DIR)/common_test.o: $(SRC_DIR)/dnn/common_test.cc
	$(CXX) $(CXXFLAGS) $^ -c -o $@

.PHONY: clean
clean:
	$(RM) $(OBJ_DIR)/* $(TARGET_LIB)
