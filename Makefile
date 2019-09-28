CXX := clang++
CUDA_DIR := /usr/local/cuda
GTEST_DIR := ./external/googletest/googletest
CXXFLAGS := -g -std=c++11 -pedantic -Wall -Wextra -fno-exceptions -fPIC -I . \
            -I $(CUDA_DIR)/include -I $(GTEST_DIR)/include
TARGET_LIB := ./lib/libcuxx.so
OBJ_DIR := ./obj
OBJS := $(OBJ_DIR)/common.o $(OBJ_DIR)/common_test.o
LDFLAGS := -L $(CUDA_DIR)/lib64 -l pthread -l cuda -l cudart -l cudnn

# The target lib.

$(TARGET_LIB): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -shared $^ -o $@

SRC_DIR := ./cuxx

# Object files.

$(OBJ_DIR)/common.o: $(SRC_DIR)/dnn/common.cc
	$(CXX) $(CXXFLAGS) $^ -c -o $@

.PHONY: test clean

# Runs tets.

TEST_BIN_DIR := ./test_bin

test: $(TEST_BIN_DIR)/gtest_main
	./test_bin/gtest_main

# The test main.

$(TEST_BIN_DIR)/gtest_main: $(OBJ_DIR)/common.o $(OBJ_DIR)/common_test.o \
                            $(GTEST_DIR)/libgtest.a $(GTEST_DIR)/libgtest_main.a
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

# Google Test.
$(GTEST_DIR)/libgtest.a $(GTEST_DIR)/libgtest_main.a:
	(cd $(GTEST_DIR) && CXX=$(CXX) cmake CMakeLists.txt && make)

# Test object files.

$(OBJ_DIR)/common_test.o: $(SRC_DIR)/dnn/common_test.cc
	$(CXX) $(CXXFLAGS) $^ -c -o $@

clean:
	$(RM) $(OBJ_DIR)/* $(TARGET_LIB) $(TEST_BIN_DIR)/* \
	$(GTEST_DIR)/libgtest.a $(GTEST_DIR)/libgtest_main.a \
	$(GTEST_DIR)/CMakeCache.txt $(GTEST_DIR)/cmake_install.cmake \
	$(GTEST_DIR)/Makefile && $(RM) -r $(GTEST_DIR)/CMakeFiles
