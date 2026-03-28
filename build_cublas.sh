#!/usr/bin/env bash
# Build cuBLAS benchmark on Linux (Colab, EC2). Detect GPU arch when possible.
set -euo pipefail
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
NVCC="${CUDA_PATH}/bin/nvcc"
if [[ ! -x "${NVCC}" ]]; then
  echo "nvcc not found at ${NVCC}. Set CUDA_PATH." >&2
  exit 1
fi

ARCH_FLAGS=""
if command -v nvidia-smi >/dev/null 2>&1; then
  # Map common compute capabilities; user can override with NVCC_ARCH_FLAGS
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
  case "${CC}" in
    60) ARCH_FLAGS="-gencode arch=compute_60,code=sm_60" ;;
    70) ARCH_FLAGS="-gencode arch=compute_70,code=sm_70" ;;
    75) ARCH_FLAGS="-gencode arch=compute_75,code=sm_75" ;;
    80) ARCH_FLAGS="-gencode arch=compute_80,code=sm_80" ;;
    86) ARCH_FLAGS="-gencode arch=compute_86,code=sm_86" ;;
    87) ARCH_FLAGS="-gencode arch=compute_87,code=sm_87" ;;
    89) ARCH_FLAGS="-gencode arch=compute_89,code=sm_89" ;;
    90) ARCH_FLAGS="-gencode arch=compute_90,code=sm_90" ;;
    120) ARCH_FLAGS="-gencode arch=compute_90,code=sm_90" ;; # Hopper 12.0 → sm_90
    *) ARCH_FLAGS="-gencode arch=compute_80,code=sm_80" ;;
  esac
fi
if [[ -z "${ARCH_FLAGS:-}" ]]; then
  ARCH_FLAGS="-gencode arch=compute_80,code=sm_80"
fi
ARCH_FLAGS="${NVCC_ARCH_FLAGS:-$ARCH_FLAGS}"

echo "Using nvcc arch: ${ARCH_FLAGS}"
"${NVCC}" -O3 -std=c++17 ${ARCH_FLAGS} -lcublas -o benchmark_cublas_fc benchmark_cublas.cu
echo "Built ./benchmark_cublas_fc"
