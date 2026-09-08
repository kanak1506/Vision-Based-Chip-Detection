# Vision-Based Metal Chip Classification & Machining Monitoring

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0%20%7C%204.8+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-Custom-lightgrey.svg)](#license)

An end-to-end, classical computer-vision pipeline for automated real-time metal chip morphology segmentation, tracking, and ISO 3685 chip form classification during CNC/manual lathe turning operations.

Based on the multi-stage spatial-temporal methodology pioneered by Filep et al., augmented with an advanced **9-layer hybrid segmentation stack** designed to handle severe workpiece specular reflections, machine vibration jitter, rotating chuck glare, and complex chip geometries (continuous ribbons, helical coils, and elemental segmental chips).

---

## System Architecture & Pipeline Phases

```
[Raw Lathe Turning Video (848x478 @ 60 FPS)]
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│ Phase 0: Video Inventory, Metadata & Leakage-Free Split │
│   • Video-wise partition (70% Train, 15% Val, 15% Test)│
└───────────────────┬────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│ Phase 1: Robust Machine Tool Tracking & Calibration    │
│   • Multi-tracker (CSRT / Template Correlation)        │
│   • Automated feed displacement & tool-tip coordinates │
└───────────────────┬────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│ Phase 2: Temporal Subsampling (~4 FPS) & ROI Cropping  │
│   • 300x300px Tool-Tip Centered ROI Crop Extraction   │
│   • Sharpness (Laplacian Var) & Motion Quality Tagging │
└───────────────────┬────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│ Phase 3: 9-Layer Classical Segmentation Stack          │
│   • Layer 0:  Dichromatic Specular Pre-Filtering       │
│   • Layer 1:  LAB Decoupling + CLAHE + Bilateral Filter│
│   • Layer 2:  Multi-Scale Frangi Curvilinear Ridges   │
│   • Layer 3:  Asymmetric Cutting Sector Masking        │
│   • Layer 4:  Adaptive Background Subtraction (MOG2)   │
│   • Layer 4b: Dense Optical Flow Motion-Coherence Gate │
│   • Layer 5:  Workpiece Specular Line & Corner Filter  │
│   • Layer 6:  Seed-Connected Multi-Hop Geodesic Tracing│
│   • Layer 7:  Orientation Closing & 1px Skeletonization│
│   • Layer 8:  HUD Telemetry & Crisp Overlay Rendering  │
└───────────────────┬────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│ Phase 3.10: Morphological Feature Extraction & ISO 3685│
│   • Curl radius, ribbon length, aspect ratio, clusters │
└────────────────────────────────────────────────────────┘
```

---

## 9-Layer Segmentation Stack (Phase 3 v2)

| Layer | Module | Description |
| :--- | :--- | :--- |
| **Layer 0** | [`src/specular_prefilter.py`](src/specular_prefilter.py) | **Dichromatic Specular Pre-Filtering**: Inpaints high-value, low-saturation specular glare patches before edge detection. |
| **Layer 1** | [`src/lab_preprocessor.py`](src/lab_preprocessor.py) | **LAB Decoupling & Denoising**: Extracts $L$-channel, applies Contrast-Limited Adaptive Histogram Equalization (CLAHE), and bilateral edge-preserving smoothing. |
| **Layer 2** | [`src/vesselness_filter.py`](src/vesselness_filter.py) | **Multi-Scale Frangi Ridge Filtering**: Computes Hessian matrix eigenvalues across scales ($\sigma \in [1, 5]\text{px}$) to isolate continuous curvilinear chip strands. |
| **Layer 3** | [`src/sector_masker.py`](src/sector_masker.py) | **Asymmetric Spatial Sector Masking**: Masks solid workpiece body, rotating spindle/chuck, lathe bed floor, and steel tool carriage while keeping active cutting and ribbon corridors open. |
| **Layer 4** | [`src/baseline_subtractor.py`](src/baseline_subtractor.py) | **Adaptive Background Modeling**: Gaussian Mixture Model (MOG2) that absorbs static machine textures while retaining dynamic foreground. |
| **Layer 4b** | [`src/flow_gate.py`](src/flow_gate.py) | **Dense Optical Flow Gating**: Computes Farneback velocity fields to discard stationary debris and vibration artifacts ($M \ge 0.8\text{ px/frame}$) while protecting the $25\text{px}$ shear root. |
| **Layer 5** | [`src/specular_filter.py`](src/specular_filter.py) | **Geometric Line & Corner Filtering**: Suppresses straight horizontal cylindrical reflection streaks ($R^2 > 0.92, |\theta| \le 16^\circ$) and rigid $90^\circ$ tool corners. |
| **Layer 6** | [`src/contour_extractor.py`](src/contour_extractor.py) | **Seed-Connected Component Tracing**: Multi-hop geodesic chaining originating at the tool tip seed ($r=25\text{px}$) along verified ejection corridors. |
| **Layer 7** | [`src/contour_extractor.py`](src/contour_extractor.py) | **1px Medial Axis Skeletonization**: Multi-directional morphological closing and Zhang-Suen thinning to extract true 1-pixel centerline spines. |
| **Layer 8** | [`src/overlay_renderer.py`](src/overlay_renderer.py) | **Real-Time HUD Overlay**: Renders clean neon green single-line chip boundaries and telemetry metrics (Cluster count, Chip area in px). |

---

## Repository Structure

```
.
├── src/                          # Modular core vision pipeline
│   ├── config.py                 # Centralized configuration & cross-platform path management
│   ├── metadata_extractor.py     # Phase 0: Video metadata reader (FPS, resolution, codec)
│   ├── dataset_splitter.py       # Phase 0: Video-wise deterministic train/val/test splitter
│   ├── frame_extractor.py        # Phase 0: Full-resolution frame extraction
│   ├── tool_tracker.py           # Phase 1: Machine tool tip tracker
│   ├── robust_tool_tracker.py    # Phase 1: Multi-tracker ensemble with motion consistency
│   ├── tracking_renderer.py      # Phase 1: Tracking QA visualization renderer
│   ├── subsample_filter.py       # Phase 2: ~4 FPS subsampling, quality tagging & 300x300 crop extraction
│   ├── specular_prefilter.py     # Phase 3 (L0): Dichromatic reflection inpainting
│   ├── lab_preprocessor.py       # Phase 3 (L1): LAB lightness + CLAHE + bilateral filtering
│   ├── canny_edge_detector.py    # Phase 3 (L2a): Adaptive Dynamic Canny edge detector
│   ├── vesselness_filter.py      # Phase 3 (L2b/c): Multi-scale Frangi curvilinear ridge filter
│   ├── sector_masker.py          # Phase 3 (L3): Universal asymmetric machining sector masking
│   ├── baseline_subtractor.py    # Phase 3 (L4): Adaptive MOG2 background subtractor
│   ├── flow_gate.py              # Phase 3 (L4b): Dense optical flow motion-coherence gate
│   ├── specular_filter.py        # Phase 3 (L5): Specular reflection line & rigid corner filter
│   ├── contour_extractor.py      # Phase 3 (L6/7): Seed-connected tracing & 1px skeletonization
│   ├── overlay_renderer.py       # Phase 3 (L8): Real-time HUD banner & chip contour overlay
│   └── morphology_extractor.py   # Phase 3.10: Morphological metric computation for ISO 3685
├── scripts/                      # Runner and QA visualization scripts
│   ├── run_phase0_metadata_and_split.py # Phase 0 batch runner
│   ├── run_phase0_extract_frames.py     # Phase 0 frame extraction runner
│   ├── run_phase1_tracker.py            # Phase 1 tool tracking batch runner
│   ├── run_phase2_subsample.py          # Phase 2 subsampling and ROI crop runner
│   ├── run_phase3_v2_pipeline.py        # Phase 3 complete 9-layer batch runner (multiprocessed)
│   ├── calibrate_tool_tip.py            # Interactive tool-tip calibration tool
│   ├── launch_tracker_gui.py            # Interactive tracker tuning GUI
│   └── view_step*.py                    # Stepwise layer visualizers (L1 through L8)
├── data/
│   ├── phase0_raw_videos/        # User video input directory (excluded via .gitignore)
│   └── metadata/                 # Dataset manifests, video inventory & split splits
│       ├── video_metadata.csv
│       ├── dataset_splits_video_wise.csv
│       ├── dataset_splits_summary.json
│       ├── tracking_results.csv
│       ├── phase2_subsample_manifest.csv
│       ├── phase3_manifest.csv
│       └── phase3_summary.json
├── docs/                         # Project documentation and specifications
├── requirements.txt              # Pinned Python package dependencies
├── .gitignore                    # Comprehensive exclusion rules for large datasets
└── README.md
```

---

## Installation & Setup

### 1. Prerequisites
* Python 3.10, 3.11, or 3.12
* `opencv-contrib-python` (required for `ximgproc` Zhang-Suen thinning)

### 2. Clone and Install
```bash
git clone https://github.com/<your-username>/vision-based-chip-classification.git
cd vision-based-chip-classification

# Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Step 1: Place Video Data
Place your raw lathe turning MP4/AVI videos into `data/phase0_raw_videos/`.

### Step 2: Phase 0 — Metadata Extraction & Video-Wise Split
Extract video properties (duration, FPS, resolution) and generate leakage-free video-wise train/val/test splits (70/15/15 ratio):
```bash
python scripts/run_phase0_metadata_and_split.py
python scripts/run_phase0_extract_frames.py
```

### Step 3: Phase 1 — Tool Tip Tracking
Track the cutting tool insert throughout each video sequence:
```bash
python scripts/run_phase1_tracker.py
```
*(Optional: Run `python scripts/launch_tracker_gui.py` to inspect and calibrate initial tool bounding boxes).*

### Step 4: Phase 2 — Temporal Subsampling & 300x300 ROI Extraction
Subsample videos at ~4 FPS (sampling every 15th frame on 60 FPS video), compute frame sharpness (Laplacian variance), and extract stabilized $300 \times 300\text{px}$ crops centered on the tool tip:
```bash
python scripts/run_phase2_subsample.py
```

### Step 5: Phase 3 — Complete 9-Layer Hybrid Segmentation Pipeline
Run the parallelized batch segmentation pipeline across all extracted crops:
```bash
python scripts/run_phase3_v2_pipeline.py
```
Outputs are written to `data/phase3_stepwise_layers/` across all 9 layers (from Layer 0 prefiltering to Layer 8 HUD overlays) and cataloged in `data/metadata/phase3_manifest.csv`.

---

## Quality Assurance & Visualization Utilities

Inspect individual layer outputs step-by-step using the dedicated visualization scripts in `scripts/`:

```bash
# View LAB decoupling & CLAHE enhancement
python scripts/view_step2_clahe.py

# View Sector Masking coverage
python scripts/view_step3.5_sector_mask.py

# View Specular reflection removal
python scripts/view_step3.7_specular_filter.py

# View Seed-connected components & 1px skeletons
python scripts/view_step3.8_contours.py

# View final HUD overlay renderings
python scripts/view_step3.9_overlays.py
```

---

## Dataset Split Methodology

To prevent temporal data leakage across sequential video frames, splits are strictly partitioned **video-wise** rather than frame-wise:

* **Training Set (70%):** Model calibration, Frangi scale tuning, and background statistics.
* **Validation Set (15%):** Hyperparameter threshold selection and tolerance tuning.
* **Test Holdout Set (15%):** Unseen video sequences for unbiased evaluation.

All split definitions and video mappings are documented in [`data/metadata/dataset_splits_video_wise.csv`](data/metadata/dataset_splits_video_wise.csv).

---

## License

This project is licensed under a custom license. See [LICENSE](LICENSE) for details (to be configured).
