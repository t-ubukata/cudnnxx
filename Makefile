CXX := clang++
CXXFLAGS := -g -shared -std=c++11 -pedantic -Wall -Wextra -fno-exceptions
SRC_DIR := ./src/cuxx
BUILD_DIR := ./build
LIB_DIR := ./lib
INCLUDE_DIR := ./src/cuxx
SRCS := $(SRC_DIR)/
OBJS := $(SRC_DIR)/
LIB := libcuxx.so

$(LIB_DIR)/$(LIB): $(OBJS)
>-$(CXX) $^ -o $@ $(CXXFLAGS)

$(BUILD_DIR)/dnn.o: $(BUILD_DIR)/dnn/common.cc
>-$(CXX) $< -c $(CXXFLAGS)

.PHONY: clean
clean:
>-$(RM) $(BUILD_DIR)/* $(LIB_DIR)/*
