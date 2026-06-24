# 🫁 MedSAM2 Liver & Multi-Focal Tumor Segmentation

Automated 3D liver and tumor segmentation on CT volumes using **MedSAM2** — the medical adaptation of SAM2's video-object tracking architecture. The pipeline requires no manual annotation: it auto-generates prompts from the CT data and propagates masks across the entire volume in both directions.

---

## Overview

This project evolved through three stages:

| Stage | File | Model | Detection Strategy |
|---|---|---|---|
| 1 — Exploration | `notebooks/tumor_detection_colab.ipynb` | MedSAM (ViT-B) | 2D, per-slice, manual bbox |
| 2 — 3D Pipeline | `medsam2_liver_tumor.py` | MedSAM2 | 3D tracking, percentile thresholding |
| 3 — Robust Detection | `medsam2_liver_tumor_kmeans.py` | MedSAM2 | 3D tracking, **K-Means clustering** |

The key upgrade from Stage 2 → 3 is the tumor prompt strategy. Percentile thresholding fails on **large tumors** (if a tumor occupies most of the liver, the bottom-25th percentile includes healthy tissue). K-Means with 3 clusters — Dark (tumor/cyst), Medium (healthy parenchyma), Bright (vessels/calcifications) — is statistically robust to this.

---

## Pipeline

```
CT Volume (.nii/.nii.gz)
        │
        ▼
  HU Windowing [-100, 400]          ← Liver-optimised window
        │
        ▼
  Slice Export (PNG/JPG)            ← Input format for MedSAM2 video predictor
        │
        ▼
  Auto-Prompt Generation (key slice)
  ├── Liver bbox  (morphological + geometric)
  └── Tumor bboxes (percentile | K-Means)
        │
        ▼
  MedSAM2 Multi-Object Tracking
  ├── Forward propagation  →
  └── Backward propagation ←
        │
        ▼
  Post-Processing
  ├── Largest Connected Component (liver)
  └── Tumor mask ∩ Liver mask (anatomical constraint)
        │
        ▼
  Outputs
  ├── liver_mask.nii.gz
  ├── tumor_mask.nii.gz
  └── overlays/ (per-slice PNG, green=liver, red=tumor)
```

---

## Installation

### 1. Clone this repo

```bash
git clone https://github.com/MananJain-dev/medsam2-liver-tumor.git
cd medsam2-liver-tumor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up MedSAM2

The scripts auto-clone and install MedSAM2 on first run. To do it manually:

```bash
git clone https://github.com/bowang-lab/MedSAM2.git
pip install -e MedSAM2
```

### 4. Download the checkpoint

The checkpoint is auto-downloaded on first run (~300 MB). To pre-download:

```bash
mkdir checkpoints
wget -O checkpoints/MedSAM2_latest.pt \
  https://github.com/bowang-lab/MedSAM2/releases/download/v1.0/MedSAM2_latest.pt
```

> **Requirements:** NVIDIA GPU with CUDA, ~4 GB VRAM minimum. Tested on CUDA 11.8+.

---

## Usage

### Stage 2 — Percentile-based detection

```bash
python medsam2_liver_tumor.py \
  --ct_path /path/to/volume-33.nii \
  --output_dir output/ \
  --checkpoint_path checkpoints/MedSAM2_latest.pt
```

### Stage 3 — K-Means detection (recommended)

```bash
python medsam2_liver_tumor_kmeans.py \
  --ct_path /path/to/volume-33.nii \
  --output_dir output/ \
  --checkpoint_path checkpoints/MedSAM2_latest.pt
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--ct_path` | *(required)* | Path to input `.nii` or `.nii.gz` CT volume |
| `--output_dir` | `medsam2_output/` | Directory for all outputs |
| `--checkpoint_path` | `checkpoints/` | Path to `MedSAM2_latest.pt` or its parent folder |

### Outputs

```
output/
├── debug_kmeans_prompt.png     ← Auto-prompt visualisation (sanity check)
├── slices/                     ← Windowed CT slices fed to MedSAM2
├── results/
│   ├── liver_mask.nii.gz
│   └── tumor_mask.nii.gz
└── overlays/
    └── overlay_XXXX.png        ← Green=liver, Red=tumor
```

---

## Prompt Generation Strategy

### Liver (Geometric + Morphological)

The liver consistently appears in the right-upper abdominal quadrant. A fixed spatial search region is applied, followed by tissue intensity filtering (HU 50–220 after windowing) and morphological closing to isolate the largest connected component.

### Tumor — Percentile Thresholding (`medsam2_liver_tumor.py`)

Extracts pixels in the bottom 25th percentile within the liver crop as tumor candidates. Fast but fragile: fails when a large tumor occupies most of the liver region.

### Tumor — K-Means Clustering (`medsam2_liver_tumor_kmeans.py`)

Applies K-Means (k=3) to pixel intensities within the liver crop:

- **Cluster 0 (Dark)** — Tumor / Cyst / Necrosis
- **Cluster 1 (Medium)** — Healthy parenchyma
- **Cluster 2 (Bright)** — Vessels / Calcifications

A contrast check (`cluster_medium_center - cluster_dark_center > 15`) prevents false positives on healthy/uniform livers. Robust to large tumors where adaptive percentile thresholds break down.

---

## Dataset

Tested on the **LiTS (Liver Tumor Segmentation) Challenge** dataset.

- Paper: *The Liver Tumor Segmentation Benchmark (LiTS)* — Bilic et al., 2019
- Download: [https://competitions.codalab.org/competitions/17094](https://competitions.codalab.org/competitions/17094)
- Format: NIfTI volumes (`.nii`), axial CT scans

---

## Notebooks

`notebooks/tumor_detection_colab.ipynb` — A self-contained Google Colab notebook for quick exploration. Uses the **original MedSAM** (ViT-B, 2D per-slice inference) with an upload-and-run interface. Good starting point before running the full 3D pipeline locally.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/medsam2-liver-tumor/blob/main/notebooks/tumor_detection_colab.ipynb)

---

## Repository Structure

```
medsam2-liver-tumor/
├── medsam2_liver_tumor.py          # Stage 2: percentile-based detection
├── medsam2_liver_tumor_kmeans.py   # Stage 3: K-Means detection (recommended)
├── notebooks/
│   └── tumor_detection_colab.ipynb # Stage 1: exploratory Colab notebook
├── checkpoints/                    # (gitignored) MedSAM2_latest.pt goes here
├── MedSAM2/                        # (gitignored) auto-cloned on first run
├── requirements.txt
├── .gitignore
└── README.md
```

---

## References

- **MedSAM2**: [https://github.com/bowang-lab/MedSAM2](https://github.com/bowang-lab/MedSAM2)
- **SAM2**: Ravi et al., *SAM 2: Segment Anything in Images and Videos*, 2024. [arXiv:2408.00714](https://arxiv.org/abs/2408.00714)
- **MedSAM**: Ma et al., *Segment Anything in Medical Images*, Nature Communications, 2024. [arXiv:2304.12306](https://arxiv.org/abs/2304.12306)
- **LiTS Dataset**: Bilic et al., *The Liver Tumor Segmentation Benchmark (LiTS)*, Medical Image Analysis, 2023.

---

## License

MIT
