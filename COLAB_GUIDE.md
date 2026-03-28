# Google Colab — step-by-step guide (clone, run PyTorch + cuBLAS)

This guide walks through **one Colab notebook** from empty session to **CSV + PNG outputs** for both the PyTorch benchmark and the compiled cuBLAS benchmark.

---

## 1. Before you start

- A **Google account** (free Colab) or **Colab Pro** if you want better GPUs.
- Your project on GitHub **or** a **zip** of the project folder (if you do not use Git).

**Repository layout you expect after clone** (names may vary):

- `benchmark_pytorch_fc.py`
- `benchmark_cublas.cu`
- `build_cublas.sh`
- `plot_fc_csv.py`
- `requirements.txt`
- `README.md` (optional)

---

## 2. Open Colab and enable GPU

1. Go to [https://colab.research.google.com](https://colab.research.google.com).
2. **File → New notebook** (or upload an `.ipynb` if you have one).
3. **Runtime → Change runtime type**
   - **Hardware accelerator:** **GPU**
   - **GPU type:** pick **A100** or **L4** if available (better for TF32 story); **T4** works but TF32 behavior differs from Ampere-class chips.
4. Click **Save**.

---

## 3. Check GPU and CUDA compiler (one cell)

Run this in the **first code cell** so you know what machine you have:

```python
!nvidia-smi
!nvcc --version
```

- If **`nvidia-smi`** shows a GPU, PyTorch can use it.
- If **`nvcc`** prints a version (e.g. CUDA 12.x), you can **compile** `benchmark_cublas.cu` on this runtime.

---

## 4. Get the code onto Colab

You can use **Git clone** (best if the repo is on GitHub) or **upload a zip**.

### Option A — Clone with Git (public repository)

In a new cell, set your repo URL and clone into `/content` (Colab’s default workspace):

```python
%cd /content
!rm -rf cmpe-258-hw1-experiments   # optional: remove old folder if re-running
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git cmpe-258-hw1-experiments
%cd cmpe-258-hw1-experiments
!ls -la
```

Replace `YOUR_USERNAME/YOUR_REPO` with your real path, for example:

`https://github.com/yuktapramodpadgaonkar-lgtm/cmpe-258-hw1-experiments.git`

After this, **`%cd`** makes the next cells run **inside the project folder**.

### Option A2 — Private repository

GitHub no longer accepts account passwords for `git clone` over HTTPS. Use a **Personal Access Token (PAT)**:

1. GitHub → **Settings → Developer settings → Personal access tokens** → create a token with **repo** scope.
2. Clone using the token in the URL (do **not** share this notebook publicly with the token embedded):

```python
%cd /content
!git clone https://YOUR_GITHUB_USERNAME:YOUR_TOKEN@github.com/YOUR_USERNAME/YOUR_REPO.git cmpe-258-hw1-experiments
%cd cmpe-258-hw1-experiments
```

Safer: use **Colab secrets** (key icon in the sidebar) to store `GITHUB_TOKEN`, then build the URL in Python without pasting the token in plain text in a shared notebook.

### Option B — No Git: upload a zip

1. On your PC, zip the project folder.
2. In Colab, click the **folder** icon (left sidebar) → **upload** the zip.
3. Unzip and `cd`:

```python
%cd /content
!unzip -o your-project.zip -d project
%cd project/your-folder-name   # adjust to the folder name inside the zip
!ls -la
```

---

## 5. Install Python dependencies

From the **project root** (where `requirements.txt` lives):

```python
%cd /content/cmpe-258-hw1-experiments   # adjust path if yours differs
!pip install -q -r requirements.txt
```

Colab often ships with `torch`; `requirements.txt` will align or upgrade packages. If you see import errors, run:

```python
!pip install -q torch numpy matplotlib
```

---

## 6. Run the PyTorch FC benchmark

Still in the project directory:

```python
%cd /content/cmpe-258-hw1-experiments
!python benchmark_pytorch_fc.py --warmup 20 --iters 100 \
  --csv results_pytorch_fc.csv --plot-prefix colab_fc
```

**Outputs** (same folder):

- `results_pytorch_fc.csv`
- `colab_fc_pytorch_fc.png`

Optional — fewer/larger sizes for a quick test:

```python
!python benchmark_pytorch_fc.py --sizes 64,256,256 --warmup 10 --iters 50 \
  --csv results_pytorch_fc.csv --plot-prefix colab_fc
```

---

## 7. Build the CUDA cuBLAS benchmark

Requires **`nvcc`** (you already checked in step 3). From project root:

```python
%cd /content/cmpe-258-hw1-experiments
!chmod +x build_cublas.sh
!./build_cublas.sh
```

If the script cannot find `nvcc`, point CUDA at the usual Colab path and retry:

```python
import os
os.environ["CUDA_PATH"] = "/usr/local/cuda"
!./build_cublas.sh
```

If **architecture** fails (rare), set the GPU architecture explicitly. Example for **sm_75** (T4):

```python
import os
os.environ["NVCC_ARCH_FLAGS"] = "-gencode arch=compute_75,code=sm_75"
!./build_cublas.sh
```

Use `!nvidia-smi --query-gpu=compute_cap --format=csv` to see your **compute capability**, then map to `sm_XX` (e.g. 8.0 → `sm_80`).

This should create an executable **`benchmark_cublas_fc`** in the current directory.

---

## 8. Run the cuBLAS benchmark

```python
%cd /content/cmpe-258-hw1-experiments
!./benchmark_cublas_fc --warmup 20 --iters 100 --csv results_cublas_fc.csv
```

**Output:** `results_cublas_fc.csv` in the project folder.

---

## 9. Plot cuBLAS CSV (optional)

```python
%cd /content/cmpe-258-hw1-experiments
!python plot_fc_csv.py --csv results_cublas_fc.csv --title "cuBLAS FC" --out cublas_fc.png
```

---

## 10. Download results to your computer

1. Open the **Files** sidebar (folder icon).
2. Navigate to `/content/cmpe-258-hw1-experiments` (or your project path).
3. Right-click each file → **Download** for:
   - `results_pytorch_fc.csv`
   - `colab_fc_pytorch_fc.png`
   - `results_cublas_fc.csv`
   - `cublas_fc.png` (if generated)

**Optional — Google Drive**

```python
from google.colab import drive
drive.mount('/content/drive')
!cp /content/cmpe-258-hw1-experiments/*.csv /content/drive/MyDrive/
!cp /content/cmpe-258-hw1-experiments/*.png /content/drive/MyDrive/
```

---

## 11. One-shot: full pipeline in order (copy-paste blocks)

**Block 1 — setup path (edit clone URL):**

```python
%cd /content
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git hw1
%cd hw1
!pip install -q -r requirements.txt
```

**Block 2 — PyTorch:**

```python
!python benchmark_pytorch_fc.py --warmup 20 --iters 100 --csv results_pytorch_fc.csv --plot-prefix colab_fc
```

**Block 3 — cuBLAS build + run:**

```python
!chmod +x build_cublas.sh && ./build_cublas.sh
!./benchmark_cublas_fc --warmup 20 --iters 100 --csv results_cublas_fc.csv
!python plot_fc_csv.py --csv results_cublas_fc.csv --title "cuBLAS FC" --out cublas_fc.png
```

---

## 12. Troubleshooting

| Problem | What to try |
|--------|-------------|
| `git: command not found` | Rare on Colab; use **upload zip** instead. |
| `Repository not found` | Wrong URL, or private repo without token. |
| `CUDA is required` | **Runtime → Change runtime type → GPU** and re-run. |
| `nvcc: not found` | PyTorch part still works; cuBLAS build needs a runtime with CUDA toolkit or manual install (heavy). |
| `failed to push` / auth | This guide is for **Colab pull/clone**, not for fixing GitHub push from your laptop. |
| Session disconnect | Colab **deletes `/content`** when the runtime restarts; **re-clone** or **re-upload** and save outputs to **Drive** often. |

---

## 13. Where to read more

- Concepts and modes: `EXPLICATION.md` (or `documents/EXPLICATION.md` if your repo moved it).
- AWS EC2: `RUNNING_COLAB_AWS.md` or `documents/RUNNING_COLAB_AWS.md`.
