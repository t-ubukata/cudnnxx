CC := clang
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

SRC_DIR := ./cudnnxx

# Object files.

$(OBJ_DIR)/common.o: $(SRC_DIR)/common.cc
	$(CXX) $(CXXFLAGS) $^ -c -o $@
$(OBJ_DIR)/op_tensor.o: $(SRC_DIR)/op_tensor.cc
	$(CXX) $(CXXFLAGS) $^ -c -o $@
$(OBJ_DIR)/convolution.o: $(SRC_DIR)/convolution.cc
	$(CXX) $(CXXFLAGS) $^ -c -o $@
$(OBJ_DIR)/activation.o: $(SRC_DIR)/activation.cc
	$(CXX) $(CXXFLAGS) $^ -c -o $@
$(OBJ_DIR)/reduce_tensor.o: $(SRC_DIR)/reduce_tensor.cc
	$(CXX) $(CXXFLAGS) $^ -c -o $@
$(OBJ_DIR)/pooling.o: $(SRC_DIR)/pooling.cc
	$(CXX) $(CXXFLAGS) $^ -c -o $@

.PHONY: test clean

TEST_BIN_DIR := ./test_bin

# Runs tets.
test: $(TEST_BIN_DIR)/gtest_main
	./test_bin/gtest_main

TEST_OBJS := $(OBJ_DIR)/common_test.o \
             $(OBJ_DIR)/op_tensor_test.o \
             $(OBJ_DIR)/convolution_test.o \
             $(OBJ_DIR)/activation_test.o \
             $(OBJ_DIR)/reduce_tensor_test.o \
             $(OBJ_DIR)/pooling_test.o \
             $(OBJ_DIR)/example_test.o


# Google Test.
GTEST_TARGET := $(GTEST_DIR)/googletest/libgtest_main.a

$(GTEST_TARGET):
	(cd $(GTEST_DIR)/googletest && CC=$(CC) CXX=$(CXX) cmake CMakeLists.txt && make)

# The test main.
$(TEST_BIN_DIR)/gtest_main: $(OBJS) $(TEST_OBJS) $(GTEST_TARGET)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@ $(GTEST_DIR)/googletest/libgtest.a

# Test object files.

$(OBJ_DIR)/common_test.o: $(SRC_DIR)/common_test.cc
	$(CXX) $(CXXFLAGS) $^ -c -o $@
$(OBJ_DIR)/op_tensor_test.o: $(SRC_DIR)/op_tensor_test.cc
	$(CXX) $(CXXFLAGS) $^ -c -o $@
$(OBJ_DIR)/convolution_test.o: $(SRC_DIR)/convolution_test.cc
	$(CXX) $(CXXFLAGS) $^ -c -o $@
$(OBJ_DIR)/activation_test.o: $(SRC_DIR)/activation_test.cc
	$(CXX) $(CXXFLAGS) $^ -c -o $@
$(OBJ_DIR)/reduce_tensor_test.o: $(SRC_DIR)/reduce_tensor_test.cc
	$(CXX) $(CXXFLAGS) $^ -c -o $@
$(OBJ_DIR)/pooling_test.o: $(SRC_DIR)/pooling_test.cc
	$(CXX) $(CXXFLAGS) $^ -c -o $@
$(OBJ_DIR)/example_test.o: $(SRC_DIR)/example_test.cc $(TARGET_LIB)
	$(CXX) $(CXXFLAGS) $(SRC_DIR)/example_test.cc -c -o $@

clean:
	$(RM) $(OBJ_DIR)/* $(TARGET_LIB) $(TEST_BIN_DIR)/* \
	$(GTEST_DIR)/googletest/libgtest.a $(GTEST_DIR)/googletest/libgtest_main.a \
	$(GTEST_DIR)/googletest/CMakeCache.txt $(GTEST_DIR)/googletest/cmake_install.cmake \
	$(GTEST_DIR)/googletest/Makefile $(GTEST_DIR)/CMakeCache.txt \
	$(GTEST_DIR)/googlemock/libgmock.a $(GTEST_DIR)/googlemock/libgmock_main.a \
	&& $(RM) -r $(GTEST_DIR)/googletest/CMakeFiles \
