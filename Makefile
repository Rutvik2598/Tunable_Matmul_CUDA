CUDA_HOME    ?= /usr/local/cuda-12.8
CUTLASS_HOME ?= /home/rithik/cutlass
NVCC         ?= $(CUDA_HOME)/bin/nvcc
ARCH         ?= sm_120

BUILD_DIR := build

NVCCFLAGS = -O3 -std=c++17 -arch=$(ARCH) \
            -I$(CUTLASS_HOME)/include \
            --expt-relaxed-constexpr \
            -Xcompiler -Wno-deprecated-declarations
LIBS = -lcublas

# bench: original 4-method comparison (default)
bench: $(BUILD_DIR)/bench

# sweep: separate retuning experiment
sweep: $(BUILD_DIR)/sweep

all: bench sweep

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/bench: bench.cu naive_kernel.cu tunable_kernel.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) bench.cu naive_kernel.cu tunable_kernel.cu -o $@ $(LIBS)

# Sweep instantiates the kernel ~400 times. --threads 0 lets nvcc parallelize.
$(BUILD_DIR)/sweep: sweep.cu naive_kernel.cu tunable_configs.inc | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --threads 0 sweep.cu naive_kernel.cu -o $@ $(LIBS)

tunable_configs.inc: gen_configs.py
	python3 gen_configs.py $@

run:       $(BUILD_DIR)/bench;   $(BUILD_DIR)/bench
sweep-run: $(BUILD_DIR)/sweep;   $(BUILD_DIR)/sweep

clean:
	rm -rf $(BUILD_DIR) tunable_configs.inc sweep_*.csv

.PHONY: all bench sweep run sweep-run clean
