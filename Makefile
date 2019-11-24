CXX := clang++
CUDA_DIR := /usr/local/cuda
GTEST_DIR := ./external/googletest
CXXFLAGS := -g -std=c++11 -Wall -Wextra -Werror -pedantic -pedantic-errors \
            -fno-exceptions -fPIC -I. \
            -I$(CUDA_DIR)/include -I$(GTEST_DIR)/googletest/include
TARGET_LIB := ./lib/libcuxx.so
OBJ_DIR := ./obj
OBJS := $(OBJ_DIR)/common.o $(OBJ_DIR)/op_tensor.o $(OBJ_DIR)/convolution.o
LDFLAGS := -L$(CUDA_DIR)/lib64 -lpthread -lcuda -lcudart -lcudnn

# The target lib.
$(TARGET_LIB): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -shared $^ -o $@

SRC_DIR := ./cuxx

# Object files.

$(OBJ_DIR)/common.o: $(SRC_DIR)/dnn/common.cc
	$(CXX) $(CXXFLAGS) $^ -c -o $@

$(OBJ_DIR)/op_tensor.o: $(SRC_DIR)/dnn/op_tensor.cc
	$(CXX) $(CXXFLAGS) $^ -c -o $@

$(OBJ_DIR)/convolution.o: $(SRC_DIR)/dnn/convolution.cc
	$(CXX) $(CXXFLAGS) $^ -c -o $@

.PHONY: test clean

TEST_BIN_DIR := ./test_bin

# Runs tets.
test: $(TEST_BIN_DIR)/gtest_main
	./test_bin/gtest_main

TEST_OBJS := $(OBJ_DIR)/common_test.o $(OBJ_DIR)/op_tensor_test.o \
             $(OBJ_DIR)/convolution_test.o

# The test main.
$(TEST_BIN_DIR)/gtest_main: $(OBJS) $(TEST_OBJS) \
                            $(GTEST_DIR)/googletest/libgtest.a \
                            $(GTEST_DIR)/googletest/libgtest_main.a
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

# Google Test.
$(GTEST_DIR)/googletest/libgtest.a $(GTEST_DIR)/googletest/libgtest_main.a:
	(cd $(GTEST_DIR)/googletest && CXX=$(CXX) cmake CMakeLists.txt && make)

# Test object files.

$(OBJ_DIR)/common_test.o: $(SRC_DIR)/dnn/common_test.cc
	$(CXX) $(CXXFLAGS) $^ -c -o $@

$(OBJ_DIR)/op_tensor_test.o: $(SRC_DIR)/dnn/op_tensor_test.cc
	$(CXX) $(CXXFLAGS) $^ -c -o $@

$(OBJ_DIR)/convolution_test.o: $(SRC_DIR)/dnn/convolution_test.cc
	$(CXX) $(CXXFLAGS) $^ -c -o $@

clean:
	$(RM) $(OBJ_DIR)/* $(TARGET_LIB) $(TEST_BIN_DIR)/* \
	$(GTEST_DIR)/googletest/libgtest.a $(GTEST_DIR)/googletest/libgtest_main.a \
	$(GTEST_DIR)/googletest/CMakeCache.txt $(GTEST_DIR)/googletest/cmake_install.cmake \
	$(GTEST_DIR)/googletest/Makefile $(GTEST_DIR)/CMakeCache.txt \
	$(GTEST_DIR)/googlemock/libgmock.a $(GTEST_DIR)/googlemock/libgmock_main.a \
	&& $(RM) -r $(GTEST_DIR)/googletest/CMakeFiles \
