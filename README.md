# CMPE-258 — Homework 1 (Excellent): FC GEMM — CUDA Cores vs Tensor Cores (TF32)

This project benchmarks a **fully connected (FC) layer** in two ways: **PyTorch** (TF32 on vs off) and **CUDA cuBLAS** (`cublasSgemm` vs `cublasGemmEx` with TF32). It measures **average GPU latency per forward** (milliseconds), **throughput** (GFLOPS from GEMM FLOPs), **sweeps** problem sizes, and produces **CSV** plus **comparison plots**.

---

## What you need

- **GPU machine** with NVIDIA drivers and CUDA usable by PyTorch.
- **For the TF32 Tensor Core path** (Mode 2), use an **Ampere-or-newer** GPU (e.g. A10G, A100, L4) for meaningful TF32 behavior. See [EXPLICATION.md](EXPLICATION.md).
- **CUDA toolkit with `nvcc`** if you build the **cuBLAS** benchmark (Linux / Colab / typical EC2 GPU AMIs).

---

## Quick start

### Python (PyTorch)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python benchmark_pytorch_fc.py --warmup 20 --iters 100 \
  --csv results_pytorch_fc.csv --plot-prefix comparison
```

Outputs:

- `results_pytorch_fc.csv`
- `comparison_pytorch_fc.png`

### CUDA cuBLAS (Linux / cloud GPU)

```bash
chmod +x build_cublas.sh
./build_cublas.sh
./benchmark_cublas_fc --warmup 20 --iters 100 --csv results_cublas_fc.csv
```

Optional plot from any CSV with the same columns:

```bash
python plot_fc_csv.py --csv results_cublas_fc.csv --title "cuBLAS FC" --out cublas_fc.png
```

### Custom sizes

```bash
python benchmark_pytorch_fc.py --sizes 64,512,512 128,1024,1024 --warmup 20 --iters 100
./benchmark_cublas_fc --sizes 64,512,512 128,1024,1024 --warmup 20 --iters 100 --csv results_cublas_fc.csv
```

---

## Repository layout

| File | Description |
|------|-------------|
| [benchmark_pytorch_fc.py](benchmark_pytorch_fc.py) | `nn.Linear` FC benchmark; TF32 off vs on; CUDA events; CSV + plots |
| [benchmark_cublas.cu](benchmark_cublas.cu) | `cublasSgemm` vs `cublasGemmEx` (TF32); CUDA events; CSV |
| [build_cublas.sh](build_cublas.sh) | Builds `benchmark_cublas_fc` with `nvcc` (auto GPU arch from `nvidia-smi`) |
| [Makefile](Makefile) | Alternative build; set `CUDA_ARCH` for your GPU |
| [plot_fc_csv.py](plot_fc_csv.py) | Plots latency and throughput from benchmark CSVs |
| [requirements.txt](requirements.txt) | Python dependencies |
| [EXPLICATION.md](EXPLICATION.md) | Assignment terminology, Mode 1 vs 2, code map |
| [RUNNING_COLAB_AWS.md](RUNNING_COLAB_AWS.md) | Google Colab and AWS EC2 setup, instance suggestions |

---

## Modes compared

| Mode | PyTorch | cuBLAS |
|------|---------|--------|
| Baseline (CUDA-core–style / TF32 off) | `torch.backends.cuda.matmul.allow_tf32 = False` | `cublasSgemm` |
| Tensor Core / TF32 (FP32 I/O) | `allow_tf32 = True` | `cublasGemmEx` + `CUBLAS_COMPUTE_32F_FAST_TF32` |

Data is **synthetic** (random or constant tensors) for **performance measurement only** — no training or test dataset.

---

## Documentation

- **Concepts and requirements walkthrough:** [EXPLICATION.md](EXPLICATION.md)
- **Colab and AWS (including quota and regions):** [RUNNING_COLAB_AWS.md](RUNNING_COLAB_AWS.md)

---

## License / course use

Produced for coursework (CMPE-258). Adapt as your instructor allows.
