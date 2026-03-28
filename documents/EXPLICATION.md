# Homework 1 — “Excellent” option: glossary and code map

This document explains **each requirement phrase** from the assignment text, the **ideas behind the experiment**, and how **each part of the provided code** maps to those ideas. It also states clearly how **Mode 1** and **Mode 2** differ.

---

## 1. The assignment text — phrase by phrase

Below, words are unpacked in the order they appear in the prompt.

| Phrase | Meaning |
|--------|---------|
| **Homework 1 — Excellent** | An optional extension beyond the base homework; grading may award an “Excellent” level if you complete it well. |
| **optional** | Not required for a passing or standard grade; extra work. |
| **students who would like to earn an Excellent rating** | Motivation: go beyond minimum expectations. |
| **Goal** | The main scientific/engineering outcome you must achieve. |
| **Measure** | Collect **numbers** (time, throughput), not only qualitative claims. |
| **explain** | Interpret results: *why* one path is faster and *when* the gap appears. |
| **performance difference** | How much faster/slower (latency, throughput), possibly as a ratio or curve over sizes. |
| **traditional CUDA-core** | Execution on NVIDIA’s **general-purpose** floating-point units inside each SM (streaming multiprocessor), used for standard FP32 `SGEMM`-style math. |
| **GEMM** | **GE**neral **M**atrix **M**ultiply: compute \(C \approx \alpha AB + \beta C\) (sizes chosen so the dominant work is the multiply-add chain). |
| **path** | A specific implementation route: which library call, which math mode, which hardware feature. |
| **modern Tensor Core** | NVIDIA’s **specialized** matrix units (on supported GPUs) that do fast fused multiply-add on small tiles, often with reduced precision *internally* while keeping FP32 storage. |
| **simple fully connected layer** | One `Linear` / FC layer: \(y = x W^T + b\). The heavy work is a GEMM (plus a small vector add for bias). |
| **Implement a complete FC model** | Use a real `nn.Linear` (PyTorch) or equivalent GEMM dimensions in CUDA so the benchmark reflects an actual layer, not a toy loop with no structure. |
| **same input sizes** | For each comparison, **B** (batch), **K** (input features), **N** (output features) must match between modes so timing differences come from the **math path**, not different shapes. |
| **compare** | Run both modes on the **same** hardware and sizes; report side-by-side metrics. |
| **Mode 1 (CUDA Core baseline): FP32 matmul with TF32 disabled** | **Storage/compute types stay FP32**, but you **turn off** the library’s permission to use **TF32** for matrix multiply, so the implementation tends to use **CUDA cores** / non-TF32 Tensor paths depending on GPU and library. |
| **Mode 2 (Tensor Core): FP32 matmul with TF32 enabled** | Still **FP32 tensors** in memory, but matmul may use **TF32** on **Tensor Cores** (on **Ampere and newer** GPUs that support it). |
| **Implement the same benchmark in CUDA C++ using cuBLAS** | Duplicate the experiment at the **library** level: one call for classic FP32 GEMM, one for **TF32 Tensor Op** compute type. |
| **Baseline: cublasSgemm (FP32)** | Standard single-precision GEMM API; canonical “CUDA-core-style” baseline in cuBLAS. |
| **Tensor Core path: cublasGemmEx or cuBLASLt using TF32 Tensor Ops (FP32 input/output, TF32 compute)** | Use `cublasGemmEx` with compute type **`CUBLAS_COMPUTE_32F_FAST_TF32`**: **reads/writes FP32**, **accumulates using TF32** on Tensor Cores where supported. |
| **Benchmark correctly** | Avoid common mistakes: cold-start effects, CPU timing of GPU work, missing synchronization. |
| **Sweep sizes** | Try **multiple** \((B, K, N)\) triples to see scaling behavior. |
| **Warm-up iterations before timing** | Run untimed iterations first so **JIT**, **memory allocation**, and **GPU clock boosting** stabilize. |
| **GPU timing (e.g., torch.cuda.Event)** | Record timestamps **on the GPU** around kernels. |
| **torch.cuda.synchronize()** | Force completion of queued GPU work before reading timers or starting CPU-side logic that assumes GPU is idle. |
| **Report average latency per forward pass (ms)** | Total GPU time for one forward / one GEMM, divided by iteration count, in **milliseconds**. |
| **over multiple iterations** | Average many timed runs to reduce noise. |
| **Compute throughput** | Typically **GFLOPS** = (GEMM FLOPs) / (time in seconds); GEMM FLOPs often taken as \(2BNK\) for the multiply-add count of the dominant matmul. |
| **clear comparison plots** | Charts (latency vs problem, throughput vs problem) comparing modes. |

**Important hardware note:** **TF32 Tensor Ops** for FP32 GEMM are meaningful on **Ampere (SM 8.0)** and **newer** datacenter/consumer GPUs in the supported configurations. **Volta (V100)** has Tensor Cores but **not** this TF32 mode; **Turing (T4)** is **not** the right generation for TF32 as defined here. Prefer **A10G, A100, L4, RTX 30/40**-class when possible.

---

## 2. What the experiment is doing (one paragraph)

You time **the same matrix multiply work** two ways: (1) **without** allowing TF32-based Tensor Core acceleration for matmul, and (2) **with** TF32 allowed. In PyTorch this is controlled by `torch.backends.cuda.matmul.allow_tf32` (and `cudnn` for convolutions). In cuBLAS you compare **`cublasSgemm`** to **`cublasGemmEx`** with **`CUBLAS_COMPUTE_32F_FAST_TF32`**. You **warm up**, time on the **GPU**, **sweep** several layer shapes, and plot **latency** and **throughput**.

---

## 3. How Mode 1 differs from Mode 2

| Aspect | Mode 1 — “CUDA-core baseline” (TF32 off) | Mode 2 — “Tensor Core / TF32” (TF32 on) |
|--------|------------------------------------------|----------------------------------------|
| **Tensor dtype** | FP32 | FP32 |
| **PyTorch switch** | `allow_tf32 = False` | `allow_tf32 = True` |
| **Typical hardware** | More work on **CUDA cores** (or non-TF32 library paths) | On supported GPUs, matmul can use **Tensor Cores** with **TF32** internal precision |
| **Numerical detail** | Full FP32 multiply-add pipeline for the core op (subject to library implementation) | **TF32**: ~10-bit mantissa internally; still **FP32** in memory |
| **cuBLAS call** | `cublasSgemm` | `cublasGemmEx` + `CUBLAS_COMPUTE_32F_FAST_TF32` |
| **Expected outcome** | Often **lower** TFLOPS on large GEMMs | Often **higher** TFLOPS on large GEMMs on Ampere+ |

---

## 4. PyTorch file (`benchmark_pytorch_fc.py`) — what each main piece does

- **`gemm_flops`**: Counts approximate FLOPs for the dominant multiply-add GEMM \(2 \times B \times N \times K\). The bias add is negligible at large sizes.
- **`set_tf32_matmul`**: Sets **both** `matmul` and `cudnn` TF32 flags so behavior matches the assignment (“FP32 matmul” family) consistently.
- **`bench_linear_once`**: Builds **`nn.Linear(K, N)`** with bias, runs **`warmup`** untimed forwards, then **`iters`** timed forwards between **`torch.cuda.Event`** records, **`synchronize()`**, and divides elapsed GPU time by **`iters`** for **average ms per forward**.
- **Throughput**: `gflops = flops / (avg_ms/1000) / 1e9`.
- **`default_sweep` / `--sizes`**: **Sweep** over several \((B, K, N)\) triples.
- **CSV + plots**: Writes **`results_pytorch_fc.csv`** and **`comparison_pytorch_fc.png`** (prefix configurable).

---

## 5. CUDA file (`benchmark_cublas.cu`) — what each main piece does

- **Problem shape**: FC forward without bias is \(C = A \times B\) with \(A\) of shape \((m \times k) = (B \times K)\), \(B\) of shape \((K \times N)\), \(C\) of shape \((B \times N)\), **column-major** as required by cuBLAS.
- **`cublasSgemm`**: Baseline FP32 GEMM.
- **`cublasGemmEx`**: Same shapes with **`CUBLAS_COMPUTE_32F_FAST_TF32`** and **`CUBLAS_GEMM_DEFAULT_TENSOR_OP`** algorithm hint.
- **Timing**: **`cudaEventRecord`** / **`cudaEventElapsedTime`** (same idea as `torch.cuda.Event`).
- **Output**: Prints to stdout and writes **`results_cublas_fc.csv`** with the **same column names** as the PyTorch CSV for easy plotting.

---

## 6. Plotting helper (`plot_fc_csv.py`)

Reads a **CSV** produced by either benchmark (columns: `mode`, `batch`, `in_features`, `out_features`, `avg_latency_ms`, `gemm_gflops`) and draws **latency** and **throughput** curves for **all modes** present in the file, aligned on **common** \((B, K, N)\) keys.

---

## 7. Deliverables checklist (aligned with “same as homework1”)

Typical homework deliverables are: **code**, **instructions to run**, **plots**, **short write-up** of results. Here you have **runnable PyTorch + cuBLAS** code, **CSV outputs**, **plot scripts**, and this **EXPLICATION** plus **`RUNNING_COLAB_AWS.md`** for environment steps. Add your own **report PDF** with figures and interpretation for submission.
