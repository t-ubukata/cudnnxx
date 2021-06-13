TARGET_BIN := test_main
BUILD_DIR := ./build
SRC_DIR := ./cudnnxx
CUDA_DIR := /usr/local/cuda

SRCS := $(shell find $(SRC_DIR) -name *.cc)
OBJS := $(SRCS:$(SRC_DIR)/%.cc=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

CPPFLAGS := -I. -I$(CUDA_DIR)/include -MMD -MP
CXXFLAGS := -g -std=c++14 -Wall -Wextra -Werror -pedantic -pedantic-errors \
            -fno-exceptions -I. -I$(CUDA_DIR)/include

LDFLAGS := -L$(CUDA_DIR)/lib64 -lcuda -lcudart -lcudnn -lgtest -lgtest_main \
           -lpthread

.PHONY: test format clean

# Runs tets.
test: $(BUILD_DIR)/$(TARGET_BIN)
	$<

$(BUILD_DIR)/$(TARGET_BIN): $(OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cc
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

format:
	clang-format -i -style=Google cudnnxx/*

clean:
	$(RM) -r $(BUILD_DIR)

-include $(DEPS)
