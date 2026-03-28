# Linux / EC2: requires CUDA toolkit (nvcc) on PATH.
CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc

# Adjust sm_XX to your GPU (e.g. sm_80 A100, sm_86 A10G, sm_89 L4, sm_75 T4)
CUDA_ARCH ?= -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86

benchmark_cublas_fc: benchmark_cublas.cu
	$(NVCC) -O3 -std=c++17 $(CUDA_ARCH) -lcublas -o $@ benchmark_cublas.cu

clean:
	rm -f benchmark_cublas_fc

.PHONY: clean
