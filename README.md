<div id="top" align="center">

# 🫁 BronchoLumen: Real-time bronchial orifice detection in video-bronchoscopy (ROCm / AMD GPU Fork)
  Yongchao Li, Marian Himstedt

  [![Journal](https://img.shields.io/badge/IJCARS-0000.00000-00000.svg)]()
  [![arXiv](https://img.shields.io/badge/arXiv-0000.00000-00000.svg)]()
  [![Download Model](https://img.shields.io/badge/Model-BronchoLumen-blue.svg)]()
  [![Hardware Support](https://img.shields.io/badge/Hardware-AMD%20ROCm%20%7C%20Windows%20%7C%20Linux%20%7C%20WSL2-red.svg)]()

</div>

## Overview

This repository is an optimized fork of **BronchoLumen** with dedicated **AMD ROCm support** on **native Windows**, **native Linux** and **WSL2** (tested on an AMD Radeon RX 7900 XTX).  
It contains a YOLO-based pipeline for detecting bronchial orifices in bronchoscopy images and videos, managed with [`uv`](https://github.com/astral-sh/uv).

> ⚠️ **This repository does not work out of the box – on purpose.**  
> There is no `pyproject.toml` in the repo, only platform templates (`windows.pyproject.toml`, `wsl.linux.pyproject.toml`).
> The templates are configured for an **AMD Radeon RX 7900 XTX (gfx1100)**. If you use a different GPU, you **must** adapt them.
> Follow [Environment Setup](#environment-setup) before running anything.

![BronchoLumen sample](./samples/frame1.jpg)

---

## 🗂 Project Structure

**Key directories & files**

- `data.yaml` – central YOLO configuration for data paths and classes
- `windows.pyproject.toml` – **template** for native Windows (AMD ROCm wheels, GPU target pinned in the file)
- `wsl.linux.pyproject.toml` – **template** for Linux / WSL2 (PyTorch ROCm wheels)
- `wsl.linux.env` – **template** for `.env` (ROCm driver flags, Linux / WSL2 only)
- `pyproject.toml`, `uv.lock`, `.env` – *your* active copies, created during setup (git-ignored, not part of the repo)
- `datasets/`
  - `train/images`, `train/labels` – training data (YOLO format)
  - `val/images`, `val/labels` – validation data
  - `test1/images`, `test1/labels` – test set 1 (used by `predict_yolo.py`)
  - `test2/images`, `test2/labels` – test set 2 (used as `split="test"`)
  - `val/vis/` – visualized ground‑truth labels
- `script/`
  - `train_yolo.py` – train the YOLO model on GPU (ROCm/CUDA:0)
  - `test_yolo.py` – evaluate the model on the test split (`split="test"`)
  - `predict_yolo.py` – run inference on test images and save results
  - `infer_video.py` – run video inference with live visualization and output video
  - helper scripts (`polygon_to_yolo.py`, `convert_yolo_labels.py`, …)
- `runs/`
  - training and inference outputs (`runs/detect/...`)
- `video/`
  - bronchoscopy videos used for inference (`*.mp4`)
- `yolov8m.pt` – base checkpoint used for training

---

<a id="environment-setup"></a>
## 🛠️ Environment Setup (with `uv` & AMD ROCm)

### 1. Find Your GPU Target

Everything below depends on your GPU architecture. Look it up first:

| Architecture | Example GPUs | LLVM target (Windows) | `HSA_OVERRIDE_GFX_VERSION` (Linux / WSL2) |
| :--- | :--- | :--- | :--- |
| **RDNA 2** | RX 6950 XT, RX 6900 XT, RX 6800 (XT) | `gfx1030` | `10.3.0` |
| **RDNA 2** | RX 6750 XT, RX 6700 XT | `gfx1031` | `10.3.0` |
| **RDNA 2** | RX 6650 XT, RX 6600 (XT) | `gfx1032` | `10.3.0` |
| **RDNA 3** | RX 7900 XTX, RX 7900 XT, RX 7900 GRE | `gfx1100` *(template default)* | `11.0.0` *(template default)* |
| **RDNA 3** | RX 7800 XT, RX 7700 XT | `gfx1101` | `11.0.0` |
| **RDNA 3** | RX 7600 (XT) | `gfx1102` | `11.0.0` |
| **RDNA 4** | RX 9070 XT, RX 9070 | `gfx1201` | `12.0.1` |
| **RDNA 4** | RX 9060 XT | `gfx1200` | `12.0.0` |

> 💡 Not sure which one you have? On Linux / WSL2 run `rocminfo | grep gfx`. All targets with Windows wheels are listed on the
> [AMD wheel index](https://stable.repo.amd.com/rocm/whl-next/) (`amd-torch-device-gfx…` packages).
> Only the RX 7900 XTX has been tested with this repository; other cards should work but are untested.

### 2. Prerequisites

- Python ≥ **3.12** (Windows template: 3.12 – 3.14)
- An up-to-date **AMD Software: Adrenalin Edition** driver (required for native Windows *and* for WSL2)
- [`uv`](https://github.com/astral-sh/uv):

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Linux / WSL2:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3a. Native Windows

1. Create your `pyproject.toml` from the template:

   ```powershell
   copy windows.pyproject.toml pyproject.toml
   ```

2. **If your GPU is not a gfx1100 card**, open `pyproject.toml` and replace **every** `gfx1100`
   (4 places, marked with `# <- adjust gfx target`) with your target from step 1. Example for an RX 7800 XT (`gfx1101`):

   ```toml
   dependencies = [
       "torch[device-gfx1101]==2.14.0+rocm10.1.0",
       "torchvision[device-gfx1101]==0.29.0a0+rocm10.1.0",
       "amd-torchvision-device-gfx1101==0.29.0a0+rocm10.1.0",
       "ultralytics>=8.4.14",
   ]

   [tool.uv.sources]
   torch = { index = "amd-rocm" }
   torchvision = { index = "amd-rocm" }
   amd-torchvision-device-gfx1101 = { index = "amd-rocm" }
   ```

   If `uv` cannot resolve these versions, check the [AMD wheel index](https://stable.repo.amd.com/rocm/whl-next/) for the
   versions currently published for your target and adjust the pins.

3. Install:

   ```powershell
   uv sync
   ```

   This installs `torch` / `torchvision` (`+rocm10.1.0`), the `rocm-sdk` components for your GPU target, and `ultralytics`.
   No `.env` file is needed on native Windows.

### 3b. Linux / WSL2

1. Create your `pyproject.toml` and `.env` from the templates:

   ```bash
   cp wsl.linux.pyproject.toml pyproject.toml
   cp wsl.linux.env .env
   ```

2. Open `.env` and set `HSA_OVERRIDE_GFX_VERSION` to the value for your GPU from step 1
   (the template uses `11.0.0` for the RX 7900 XTX):

   ```bash
   HSA_ENABLE_DXG_DETECTION=1
   HSA_OVERRIDE_GFX_VERSION=11.0.0
   ```

   > 💡 **WSL2 Note:** `HSA_ENABLE_DXG_DETECTION=1` bridges the user-space ROCm runtime with `/dev/dxg` and is mandatory for all
   > generations under WSL2. On native Linux it is not needed.

3. Make `uv` load `.env` for every `uv run` in this shell (or add the line to your `~/.bashrc`):

   ```bash
   export UV_ENV_FILE=.env
   ```

   Alternatively pass it per command: `uv run --env-file .env ...`

4. Install:

   ```bash
   uv sync
   ```

   This installs `torch` / `torchvision` from the PyTorch ROCm index (`rocm7.14`) and `ultralytics`.
   The GPU is selected at runtime via `.env`, so the Linux template itself needs no GPU-specific changes.

> ℹ️ `uv.lock` is **not** committed, because the resolved packages depend on the platform and the GPU target.
> It is created locally on your first `uv sync` and keeps your own environment reproducible from then on.

### 4. Verify GPU Detection

Verify that PyTorch identifies the AMD GPU:

```bash
uv run python -c "import torch; print('CUDA/ROCm Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

**Expected output** (the device name shows your GPU):
```text
CUDA/ROCm Available: True
Device: AMD Radeon RX 7900 XTX
```

If it prints `False`, double-check the gfx target in `pyproject.toml` (Windows) or `HSA_OVERRIDE_GFX_VERSION` in `.env`
and that `.env` is actually loaded (Linux / WSL2).

---

## 📁 Data Layout & Configuration

The project references data through `data.yaml`:

```yaml
path: /path/to/BronchoLumen  # Absolute path to repository root (adjust to your machine)
train: datasets/train/images
val:   datasets/val/images
test:  datasets/test2/images

nc: 1
names:
  - orifice
```

**Ultralytics Settings Check**  
Ensure Ultralytics looks into the current project directory rather than default cache locations:

```bash
uv run yolo settings datasets_dir="/path/to/BronchoLumen/datasets"
```

---

## 🚀 Training

### Quick Validation Run (coco8)

To test the complete forward/backward pass and ROCm kernel dispatch without full datasets:

```bash
uv run yolo detect train data=coco8.yaml model=yolov8m.pt epochs=1 imgsz=640 device=0
```

### Start BronchoLumen Training

Once dataset images and annotations are placed in `datasets/train` and `datasets/val`:

```bash
uv run python script/train_yolo.py
```

Outputs will be saved to:
- `runs/detect/yolov8m-retrain/`
  - `weights/best.pt`
  - `weights/last.pt`
  - Training metrics, PR curves, and confusion matrices.

> ⚠️ **Note on AMP (Automatic Mixed Precision):**  
> If an AMP anomaly check triggers an error on consumer ROCm cards, Ultralytics automatically falls back to full **FP32** precision.

> ⚠️ **Note on VRAM:**  
> `batch=16` in `script/train_yolo.py` was chosen for the 24 GB of the RX 7900 XTX. On GPUs with less VRAM (e.g. 8–16 GB),
> lower `batch` (e.g. to 8 or 4) if training fails with an out-of-memory error.

---

## 📊 Evaluation (`split="test"`)

Evaluate checkpoints on the dedicated test split (`datasets/test2`):

```bash
uv run python script/test_yolo.py
```

---

## 🖼️ Batch Inference on Test Images

Run inference over test directories and write predicted boxes with confidence scores:

```bash
uv run python script/predict_yolo.py
```

Outputs are saved under `runs/detect/predict_test/`.

---

## 🎥 Video Inference

Run real-time video inference with annotated frame generation:

```bash
uv run python script/infer_video.py
```

- Press **`q`** to terminate the stream.
- Annotated output is saved to the `video/` directory.

---

## 🧪 Visualizing Ground‑Truth Labels

Render ground-truth annotations over validation images:

```bash
uv run python datasets/val/visualize_yolo_val.py
```

Visualized images will be written to `datasets/val/vis/`.

---

## ✅ Command Cheat Sheet

| Task | Command |
| :--- | :--- |
| **Create Config (Windows)** | `copy windows.pyproject.toml pyproject.toml`, then adjust the gfx target |
| **Create Config (Linux / WSL2)** | `cp wsl.linux.pyproject.toml pyproject.toml` and `cp wsl.linux.env .env`, then adjust `.env` |
| **Load `.env` (Linux / WSL2)** | `export UV_ENV_FILE=.env` |
| **Sync Environment** | `uv sync` |
| **Verify ROCm Detection** | `uv run python -c "import torch; print(torch.cuda.is_available())"` |
| **Smoke Test (coco8)** | `uv run yolo detect train data=coco8.yaml model=yolov8m.pt epochs=1 device=0` |
| **Run Training** | `uv run python script/train_yolo.py` |
| **Run Evaluation** | `uv run python script/test_yolo.py` |
| **Run Image Inference** | `uv run python script/predict_yolo.py` |
| **Run Video Inference** | `uv run python script/infer_video.py` |

Happy training and analysis! 🧪📊🫁