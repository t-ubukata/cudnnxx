CUDA_DIR := /usr/local/cuda
CXXFLAGS := -g -std=c++14 -Wall -Wextra -Werror -pedantic -pedantic-errors \
            -fno-exceptions -I. -I$(CUDA_DIR)/include
OBJ_DIR := ./obj
LDFLAGS := -L$(CUDA_DIR)/lib64 -lcuda -lcudart -lcudnn -lgtest -lgtest_main \
           -lpthread

SRC_DIR := ./cudnnxx

.PHONY: test format clean

TEST_BIN_DIR := ./test_bin

# Runs tets.
test: $(TEST_BIN_DIR)/test_main
	./test_bin/test_main

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
$(OBJ_DIR)/divisive_normalization_test.o: $(SRC_DIR)/divisive_normalization_test.cc
	$(CXX) $^ $(CXXFLAGS) -c -o $@
$(OBJ_DIR)/ctc_loss_test.o: $(SRC_DIR)/ctc_loss_test.cc
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
             $(OBJ_DIR)/divisive_normalization_test.o \
             $(OBJ_DIR)/ctc_loss_test.o \
             $(OBJ_DIR)/example_test.o

# The test main.
$(TEST_BIN_DIR)/test_main: $(OBJS) $(TEST_OBJS) $(GTEST_TARGET)
	$(CXX) $^ $(LDFLAGS) -o $@

format:
	clang-format -i -style=Google cudnnxx/*

clean:
	$(RM) $(OBJ_DIR)/* $(TARGET_LIB) $(TEST_BIN_DIR)/* \
