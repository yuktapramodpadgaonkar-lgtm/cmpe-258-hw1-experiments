# Running the FC GEMM benchmark on Google Colab and AWS EC2

This project has two parts:

1. **PyTorch** — `benchmark_pytorch_fc.py` (TF32 on vs off, GPU events, sweep, plots).
2. **CUDA cuBLAS** — `benchmark_cublas.cu` compiled with `nvcc` (see `build_cublas.sh` or `Makefile`).

---

## Which GPU / instance to use

**TF32 Tensor Ops** (the assignment’s “Mode 2”) are defined for **Ampere (SM 8.0) and newer** in the usual PyTorch/cuBLAS stacks. For a **clear** CUDA-core vs TF32 Tensor Core story:

| Environment | Suggestion | Notes |
|-------------|------------|--------|
| **AWS** | **`g5.xlarge`** (1× **NVIDIA A10G**, 24 GB) | Cost-effective, **Ampere**, **TF32**. Step up to **`g5.2xlarge`** if you need more CPU/RAM for preprocessing. |
| **Higher throughput / A100** | **`p4d.24xlarge`** or **`p5`** family | **A100 / H100** class; more expensive. |
| **Avoid for TF32 story** | **`p3.*` (V100)** | Tensor Cores exist but **not** the same **TF32** mode as Ampere+. |
| **Turing T4 (`g4dn`)** | Optional | **TF32** is **not** the right generation match for this assignment; results may **not** match course expectations. |

**Recommendation for this homework:** start with **`g5.xlarge`** on Ubuntu **Deep Learning AMI** (or plain Ubuntu + CUDA toolkit).

---

## Google Colab

### Runtime

1. Open a new notebook: [https://colab.research.google.com](https://colab.research.google.com)
2. **Runtime → Change runtime type → Hardware accelerator: GPU** (pick **A100** or **T4** if offered; **A100** preferred for TF32).

### Install dependencies

In a cell:

```python
!pip install -q torch numpy matplotlib
```

Upload the project files (`benchmark_pytorch_fc.py`, `plot_fc_csv.py`, …) via the file sidebar, or clone from your repo.

### Run PyTorch benchmark

```python
!python benchmark_pytorch_fc.py --warmup 20 --iters 100 --csv results_pytorch_fc.csv --plot-prefix colab_fc
```

Outputs:

- `results_pytorch_fc.csv`
- `colab_fc_pytorch_fc.png`

### Optional: build and run CUDA benchmark on Colab

Colab **usually** has `nvcc`. From the folder that contains `benchmark_cublas.cu`:

```python
!chmod +x build_cublas.sh
!./build_cublas.sh
!./benchmark_cublas_fc --warmup 20 --iters 100 --csv results_cublas_fc.csv
```

If `nvcc` is missing, use Colab’s CUDA path (often `/usr/local/cuda`):

```python
import os
os.environ["CUDA_PATH"] = "/usr/local/cuda"
```

Then re-run `build_cublas.sh`.

Plot:

```python
!python plot_fc_csv.py --csv results_cublas_fc.csv --title "cuBLAS FC" --out cublas_fc.png
```

### Download results

Use the Colab file browser to download `*.csv` and `*.png`.

---

## AWS EC2 (Ubuntu, g5.xlarge)

### 1. Launch instance

1. AWS Console → **EC2 → Launch instance**
2. **AMI:** *Deep Learning OSS Nvidia Driver AMI GPU PyTorch* (Ubuntu) **or** Ubuntu 22.04 + you install drivers/CUDA manually.
3. **Instance type:** **`g5.xlarge`**
4. **Storage:** 30+ GiB gp3 is usually enough
5. **Security group:** allow **SSH (22)** from your IP
6. **Key pair:** create or select a `.pem` for SSH

### 2. Connect

```bash
ssh -i /path/to/your.pem ubuntu@<PUBLIC_IP>
```

### 3. System packages (if not using DLAMI with everything preinstalled)

On a minimal Ubuntu GPU image you typically:

```bash
sudo apt update
sudo apt install -y build-essential python3-pip python3-venv git
```

Install a **GPU driver** and **CUDA toolkit** that match your stack. On NVIDIA’s or Ubuntu’s instructions, or use **NVIDIA CUDA toolkit** meta-package for your distro. For DLAMI, `nvcc` and drivers are often already present — check:

```bash
nvidia-smi
nvcc --version
```

### 4. Python venv and PyTorch

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you need a specific CUDA wheel for PyTorch, follow [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for the **Linux + pip + CUDA** line matching your driver.

### 5. Run PyTorch benchmark

From the project directory:

```bash
source .venv/bin/activate
python benchmark_pytorch_fc.py --warmup 20 --iters 100 --csv results_pytorch_fc.csv --plot-prefix aws_fc
```

### 6. Build and run cuBLAS benchmark

```bash
chmod +x build_cublas.sh
./build_cublas.sh
./benchmark_cublas_fc --warmup 20 --iters 100 --csv results_cublas_fc.csv
```

If the script picks the wrong **SM** architecture, set explicitly, e.g. for **A10G (8.6)**:

```bash
export NVCC_ARCH_FLAGS="-gencode arch=compute_86,code=sm_86"
./build_cublas.sh
```

### 7. Plots

```bash
python plot_fc_csv.py --csv results_pytorch_fc.csv --title "PyTorch FC" --out pytorch_fc.png
python plot_fc_csv.py --csv results_cublas_fc.csv --title "cuBLAS FC" --out cublas_fc.png
```

### 8. Copy results to your laptop

```bash
scp -i /path/to/your.pem ubuntu@<PUBLIC_IP>:~/Assignment1.1/*.csv .
scp -i /path/to/your.pem ubuntu@<PUBLIC_IP>:~/Assignment1.1/*.png .
```

---

## Custom size sweep

Same syntax for PyTorch and cuBLAS:

```bash
python benchmark_pytorch_fc.py --sizes 64,512,512 128,1024,1024 --warmup 20 --iters 100
./benchmark_cublas_fc --sizes 64,512,512 128,1024,1024 --warmup 20 --iters 100 --csv results_cublas_fc.csv
```

---

## Troubleshooting

| Issue | What to do |
|-------|------------|
| `CUDA is required` | Select a GPU runtime / GPU instance; run `nvidia-smi`. |
| No TF32 speedup | Confirm **Ampere+** (`nvidia-smi` shows A10G, A100, etc.). |
| `nvcc` not found | Install CUDA toolkit or use DLAMI; set `CUDA_PATH`. |
| cuBLAS compile error on `CUBLAS_COMPUTE_32F_FAST_TF32` | CUDA toolkit **too old**; use **CUDA 11.0+** / recent driver + dev packages. |
