CUDA_DIR := /usr/local/cuda
GTEST_DIR := ./external/googletest
CXXFLAGS := -g -std=c++11 -Wall -Wextra -Werror -pedantic -pedantic-errors \
            -fno-exceptions -I. \
            -I$(CUDA_DIR)/include -I$(GTEST_DIR)/googletest/include
OBJ_DIR := ./obj
LDFLAGS := -L$(CUDA_DIR)/lib64 -lpthread -lcuda -lcudart -lcudnn

SRC_DIR := ./cudnnxx

.PHONY: test format clean

TEST_BIN_DIR := ./test_bin

# Runs tets.
test: $(TEST_BIN_DIR)/gtest_main
	./test_bin/gtest_main

# Test object files.
$(OBJ_DIR)/common_test.o: $(SRC_DIR)/common_test.cc
	$(CXX) $^ $(CXXFLAGS) -c -o $@
$(OBJ_DIR)/op_tensor_test.o: $(SRC_DIR)/op_tensor_test.cc
	$(CXX) $^ $(CXXFLAGS) -c -o $@
$(OBJ_DIR)/convolution_test.o: $(SRC_DIR)/convolution_test.cc
	$(CXX) $^ $(CXXFLAGS) -c -o $@
$(OBJ_DIR)/activation_test.o: $(SRC_DIR)/activation_test.cc
	$(CXX) $^ $(CXXFLAGS) -c -o $@
$(OBJ_DIR)/reduce_tensor_test.o: $(SRC_DIR)/reduce_tensor_test.cc
	$(CXX) $^ $(CXXFLAGS) -c -o $@
$(OBJ_DIR)/pooling_test.o: $(SRC_DIR)/pooling_test.cc
	$(CXX) $^ $(CXXFLAGS) -c -o $@
$(OBJ_DIR)/dropout_test.o: $(SRC_DIR)/dropout_test.cc
	$(CXX) $^ $(CXXFLAGS) -c -o $@
$(OBJ_DIR)/rnn_test.o: $(SRC_DIR)/rnn_test.cc
	$(CXX) $^ $(CXXFLAGS) -c -o $@
$(OBJ_DIR)/example_test.o: $(SRC_DIR)/example_test.cc
	$(CXX) $^ $(CXXFLAGS) -c -o $@

TEST_OBJS := $(OBJ_DIR)/common_test.o \
             $(OBJ_DIR)/op_tensor_test.o \
             $(OBJ_DIR)/convolution_test.o \
             $(OBJ_DIR)/activation_test.o \
             $(OBJ_DIR)/reduce_tensor_test.o \
             $(OBJ_DIR)/pooling_test.o \
             $(OBJ_DIR)/dropout_test.o \
             $(OBJ_DIR)/rnn_test.o \
             $(OBJ_DIR)/example_test.o

# Google Test.
GTEST_TARGET := $(GTEST_DIR)/googletest/libgtest_main.a

$(GTEST_TARGET):
	(cd $(GTEST_DIR)/googletest && CC=$(CC) CXX=$(CXX) cmake CMakeLists.txt && make)

# The test main.
$(TEST_BIN_DIR)/gtest_main: $(OBJS) $(TEST_OBJS) $(GTEST_TARGET)
	$(CXX) $^ $(GTEST_DIR)/googletest/libgtest.a $(LDFLAGS) -o $@

format:
	clang-format -i -style=Google cudnnxx/*

clean:
	$(RM) $(OBJ_DIR)/* $(TARGET_LIB) $(TEST_BIN_DIR)/* \
	$(GTEST_DIR)/googletest/libgtest.a $(GTEST_DIR)/googletest/libgtest_main.a \
	$(GTEST_DIR)/googletest/CMakeCache.txt $(GTEST_DIR)/googletest/cmake_install.cmake \
	$(GTEST_DIR)/googletest/Makefile $(GTEST_DIR)/CMakeCache.txt \
	$(GTEST_DIR)/googlemock/libgmock.a $(GTEST_DIR)/googlemock/libgmock_main.a \
	&& $(RM) -r $(GTEST_DIR)/googletest/CMakeFiles \
