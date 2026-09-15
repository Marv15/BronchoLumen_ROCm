<div id="top" align="center">

# 🫁 BronchoLumen: Real-time bronchial orifice detection in video-bronchoscopy (ROCm / AMD GPU Fork)
  Yongchao Li, Marian Himstedt

  [![Journal](https://img.shields.io/badge/IJCARS-0000.00000-00000.svg)]()
  [![arXiv](https://img.shields.io/badge/arXiv-0000.00000-00000.svg)]()
  [![Download Model](https://img.shields.io/badge/Model-BronchoLumen-blue.svg)]()
  [![Hardware Support](https://img.shields.io/badge/Hardware-AMD%20ROCm%207.x%20%7C%20WSL2-red.svg)]()

</div>

## Overview

This repository is an optimized fork of **BronchoLumen** with dedicated **AMD ROCm support** (tested on AMD Radeon RX 7900 XTX via WSL2 / Native Linux).  
It contains a YOLO-based pipeline for detecting bronchial orifices in bronchoscopy images and videos, managed with reproducible dependency locking via [`uv`](https://github.com/astral-sh/uv).

![BronchoLumen sample](./samples/frame1.jpg)

---

## 🗂 Project Structure

**Key directories & files**

- `data.yaml` – central YOLO configuration for data paths and classes
- `.env` – hardware and driver detection flags for AMD ROCm / WSL2
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

## 🛠️ Environment Setup (with `uv` & AMD ROCm)

### 1. Prerequisites

- Linux or **WSL2 on Windows** (WSL2 requires the latest AMD Adrenalin host drivers)
- Python ≥ **3.12**
- [`uv`](https://github.com/astral-sh/uv) installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Configure Environment Variables (`.env`)

Create a `.env` file in the project root to enable driver discovery and set your GPU architecture:

```bash
cat << 'EOF' > .env
HSA_ENABLE_DXG_DETECTION=1
HSA_OVERRIDE_GFX_VERSION=11.0.0
EOF
```

#### AMD Hardware Architecture Reference

Set `HSA_OVERRIDE_GFX_VERSION` according to your GPU generation:

| Architecture | Example GPUs | Target | `.env` Setting |
| :--- | :--- | :--- | :--- |
| **RDNA 2** | RX 6950 XT, RX 6900 XT, RX 6800 (XT), RX 6700 XT | `gfx1030` | `HSA_OVERRIDE_GFX_VERSION=10.3.0` |
| **RDNA 3** | RX 7900 XTX, RX 7900 XT, RX 7800 XT, RX 7700 XT | `gfx1100` | `HSA_OVERRIDE_GFX_VERSION=11.0.0` |
| **RDNA 4** | RX 9070 XT, RX 9070 | `gfx1200` | `HSA_OVERRIDE_GFX_VERSION=12.0.0` |

> 💡 **WSL2 Note:** `HSA_ENABLE_DXG_DETECTION=1` bridges the user-space ROCm runtime with `/dev/dxg` and is mandatory for all generations under WSL2.

### 3. Install Dependencies

Install the locked ROCm PyTorch stack and Ultralytics directly from `pyproject.toml` and `uv.lock`:

```bash
uv sync
```

This installs:
- `torch` & `torchvision` (`+rocm7.14`)
- `rocm-sdk` components (including device support for GFX10, GFX11, and GFX12)
- `ultralytics`

### 4. Verify GPU Detection

Verify that PyTorch identifies the AMD GPU:

```bash
uv run python -c "import torch; print('CUDA/ROCm Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

**Expected output:**
```text
CUDA/ROCm Available: True
Device: AMD Radeon RX 7900 XTX
```

---

## 📁 Data Layout & Configuration

The project references data through `data.yaml`:

```yaml
path: /home/marvi/projects/BronchoLumen  # Absolute path to repository root
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
uv run yolo settings datasets_dir="/home/marvi/projects/BronchoLumen/datasets"
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
> If an AMP anomaly check triggers an error on consumer ROCm cards, Ultralytics automatically falls back to full **FP32** precision, which runs reliably given the 24 GB VRAM headroom.

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
| **Sync Environment** | `uv sync` |
| **Verify ROCm Detection** | `uv run python -c "import torch; print(torch.cuda.is_available())"` |
| **Smoke Test (coco8)** | `uv run yolo detect train data=coco8.yaml model=yolov8m.pt epochs=1 device=0` |
| **Run Training** | `uv run python script/train_yolo.py` |
| **Run Evaluation** | `uv run python script/test_yolo.py` |
| **Run Image Inference** | `uv run python script/predict_yolo.py` |
| **Run Video Inference** | `uv run python script/infer_video.py` |

Happy training and analysis! 🧪📊🫁