# 🔧 Vision-Based Chip Detection — End-to-End Preprocessing & Pipeline Checklist

> Comprehensive checklist integrating the experimental methodology from **Filep et al. (2024)** (*Image-based chip detection during turning*) and your multi-material project goals.
> Cross off each item with `[x]` as you complete it.

---

## Phase 0 — Video Inventory, Metadata Audit & Video-Wise Hold-Out Splitting

- [ ] **Rename videos systematically** — Map raw WhatsApp filenames to standardized experiment IDs (e.g., `Al_0.5mm_280RPM_v1.mp4`, `Cu_0.75mm_1120RPM.mp4`, `Steel_0.5mm_710RPM.mp4`).
  - *Aluminium (Videos 1–7)*: Continuous ribbons & tubular coils
  - *Copper (Videos 8–12)*: Ductile continuous & spiral coils
  - *Mild Steel (Videos 13–28)*: Segmented C-shaped & oxidized colored chips
- [x] **Extract video metadata** — Verified resolution ($848 \times 478$), frame rate ($60\text{ FPS}$), total frame count (62,301 frames), and codec integrity across all 27 raw videos (`data/metadata/video_metadata.csv`).
- [x] **Stratified Video-Wise Hold-Out Splitting (In Metadata)** — Defined video-wise split mapping upfront across materials to prevent temporal/background leakage without prematurely scattering files (`data/metadata/dataset_splits_video_wise.csv`, `data/metadata/dataset_splits_summary.json`):
  - **Aluminium (18,089 frames)**: 5 Train vids (12,714 frames, 70.3%) | 1 Val vid (2,519 frames, 13.9%) | 1 Test vid (2,856 frames, 15.8%)
  - **Copper (12,316 frames)**: 4 Train vids (8,705 frames, 70.7%) | 1 Val vid (1,920 frames, 15.6%) | 1 Test vid (1,691 frames, 13.7%)
  - **Mild Steel (31,896 frames)**: 10 Train vids (22,555 frames, 70.7%) | 2 Val vids (4,755 frames, 14.9%) | 2 Test vids (4,586 frames, 14.4%)
  - **Overall (62,301 frames)**: 19 Train vids (43,974 frames, 70.6%) | 4 Val vids (9,194 frames, 14.8%) | 4 Test vids (9,133 frames, 14.7%)
- [x] **Video Frame Extraction** — Extracted and decompressed all 62,301 frames across 27 videos into structured directories (`data/phase0_extracted_frames/<video_name>/frame_xxxxxx.jpg`) in ~3.2 min (`data/metadata/frame_extraction_summary.json`).

---

## Phase 1 — Tool Tracking & Dynamic ROI (Filep et al. Method)

- [x] **Initialize CSRT Tracker on Tool Tip** — Calibrated tool tip coordinates $(x_t, y_t)$, bounding boxes, and offsets across all 27 videos (`data/metadata/tracker_config.json`).
- [x] **Construct Dynamic ROI Window** — Centered an expanded $300 \times 300\text{ px}$ bounding ROI window around tracked tool tip coordinates $(x_t, y_t)$ to eliminate lathe bed clutter.
- [x] **Generate Initial Tracking Results** — Logged **62,301 tool trajectory points** across 27 videos in `data/metadata/tracking_results.csv` and `data/metadata/tracking_summary.json` at 81.3 FPS.
- [x] **Render Tracked Frames Dataset** — Rendered and saved all 62,301 frames with Blue Dot tool tip, CSRT BBox, and dynamic $300\times300\text{ px}$ ROI overlays into `data/phase1_tracked_frames/<video_name>/frame_xxxxxx.jpg` (`data/metadata/phase1_rendering_summary.json`).

---

## Phase 2 — Frame Extraction, Full Quality Retention & Split-Aware Structuring

- [x] **Tool Presence Verification (Stage 1)** — Verified active cutting tool presence; retained 3,820 active machining crops while segregating 345 non-cutting/off-target frames (`data/metadata/phase2_subsample_manifest.csv`).
- [x] **Sub-sample Machining Frames at ~4 FPS (Stage 2)** — Subsampled every 15th frame across 62,301 raw frames into 4,165 curated $300\times300\text{ px}$ ROI crops organized by video in `data/phase2_roi_crops/`.
- [x] **Continuous Sharpness & Motion Quality Tagging (Stage 3)** — Computed and logged per-frame `laplacian_var` and `motion_score` directly into the master manifest.
- [x] **Automated Pre-Run & Post-Run Cleanup** — Integrated auto-cleanup in `src/subsample_filter.py` to prevent stale/orphaned directories when tracking updates.
- [x] **Generate Master Phase 2 Manifest & Clean Output** — 4,165 curated crops cataloged in `data/metadata/phase2_subsample_manifest.csv` and `data/metadata/phase2_summary.json`.

---

## 🎯 Phase 3 — Strict 7-Layer Dynamic Chip Segmentation Pipeline (Filep et al. Methodology)
> **Note (Robustness Upgrade):** Phase 3 has been upgraded to the **Phase 3 v2 9-Layer Hybrid Pipeline** (documented in detail in [`planv2.md`](planv2.md) and [`plan_v2_summary.md`](plan_v2_summary.md)) to integrate multi-scale Frangi vesselness, dense optical flow motion gating, adaptive MOG2 background subtraction, and dichromatic specular pre-filtering. The full 4,165-crop batch execution has completed (`data/phase3_stepwise_layers/`).

- [x] **Layer 1: LAB Color Decoupling, Equalization & Edge-Preserving Denoising** (`src/lab_preprocessor.py`):
  - Converts BGR to CIELAB, separating Luminance ($L$) from Chrominance ($A, B$).
  - Applies CLAHE (`clipLimit=2.5`, $8\times8$ grid) to normalize harsh metallic reflections and deep tool shadow gradients.
  - Applies Bilateral Filtering ($d=5, \sigma_{\text{color}}=50, \sigma_{\text{space}}=50$) on the $L$ channel to remove shop floor texture grain while preserving razor-sharp 1px chip boundary contrast.
- [x] **Layer 2: Adaptive Canny Edge Extraction** (`src/canny_edge_detector.py`):
  - Computes dynamic median-based high/low gradient thresholds ($0.66 \cdot v \leftrightarrow 1.33 \cdot v$).
  - Extracts 1-pixel binary physical edge maps self-calibrated to each material's surface reflectivity.
- [x] **Layer 3: Universal Asymmetric Machining Sector Masking** (`src/sector_masker.py`):
  - Enforces physical zero-chip spatial boundaries:
    - *Solid Workpiece Body*: Suppresses rotating cylinder body ($x \le \text{tip\_x}-5, y \le \text{tip\_y}-12$).
    - *Lower Floor Tray*: Suppresses machine floor tray clutter ($x \le \text{tip\_x}-20, y \ge \text{tip\_y}+20$).
    - *Tool Block & Carriage*: Suppresses carriage below cutting height ($x \ge \text{tip\_x}+30, y \ge \text{tip\_y}+15$).
    - *Tool Shank*: Suppresses solid steel tool shank body ($x \in [\text{tip\_x}-10, \text{tip\_x}+40], y \ge \text{tip\_y}+25$).
  - Unmasks the full **Universal Cutting Envelope**:
    - *Rightward Ribbon Flow*: $x \in [\text{tip\_x}-10, \text{tip\_x}+150], y \in [\text{tip\_y}-60, \text{tip\_y}+15]$ (air-zone ribbon corridor).
    - *Downward/Leftward Helical Flow*: $x \in [\text{tip\_x}-120, \text{tip\_x}+10], y \in [\text{tip\_y}-10, \text{tip\_y}+130]$ (helical spring curls).
- [x] **Layer 4: Adaptive Background Subtraction (MOG2)** (`src/baseline_subtractor.py`):
  - Continuous adaptive Gaussian Mixture Model ($E_{\text{dynamic}} = E_{\text{masked}} \land \text{fg\_mask}$) to eliminate static bed ways and settled floor debris.
- [x] **Layer 5: Workpiece Specular Reflection Line & Corner Filtering** (`src/specular_filter.py`):
  - Computes orthogonal PCA linearity residual distance ($\le 1.6\text{ px}$) and orientation on edge segments.
  - Erases horizontal specular glare streaks on the rotating cylinder ($R^2 > 0.92, |\theta| \le 16^\circ$) and rigid $90^\circ$ tool corners while strictly protecting curved chip strands.
- [x] **Layer 6: Tool-Tip Seed-Connected Component Tracing** (`src/contour_extractor.py`):
  - Evaluates 8-way graph connectivity directly from the cutting insert contact seed $(150, 150)$ outwards into active chip streams ($\text{max\_gap\_ribbon} = 40.0\text{px}$).
  - Retains immediate shear proximity fragments ($\le 25\text{px}$) while discarding disconnected shop clutter.
- [x] **Layer 7: Directional Ribbon Continuity & 1-Pixel Spine Skeletonization** (`src/contour_extractor.py` & `src/overlay_renderer.py`):
  - Applies 8-direction orientation-adaptive closing and Zhang-Suen thinning (`cv2.ximgproc.thinning`, `min_contour_length = 4\text{px}`) to produce a continuous 1-pixel green spine.
  - Renders high-contrast segmented overlays with live cluster count, total area, and telemetry HUD metrics.
- [x] **Step 3.10: Morphology Feature Extraction & ISO 3685 Classification Engine** (`src/morphology_extractor.py`):
  - Compute circularity ($C = \frac{4\pi A}{P^2}$), bounding box aspect ratio, solidity, and curl count.
  - Map extracted features to ISO 3685 chip categories (*Continuous Ribbon, Helical Spring, C-Shaped, Short Arc, Washer-Type*).

---

## Phase 4 — Lighting & Thermal Color Space Normalization

- [ ] **Multi-Color Space Extraction** — Convert ROI crops into:
  - **BGR**: Standard deep learning input (YOLOv8/v11).
  - **HSV**: Isolate Hue ($H$) and Saturation ($S$) channels for temper color segmentation (Silver $\rightarrow$ Straw Gold $\rightarrow$ Purple $\rightarrow$ Dark Blue).
  - **LAB**: Perceptually uniform color space for thermal shift metrics.
- [ ] **Exposure Normalization via CLAHE** — Apply Contrast-Limited Adaptive Histogram Equalization (`cv2.createCLAHE()`) on Value ($V$) channel to normalize shop lighting variations.
- [ ] **Thermal Oxidation Color Masking** — Quantify percentage of high-friction blue/purple thermal temper on chips to flag tool wear.

---

## Phase 5 — Material-Specific Tuning & Benchmark Flags

- [ ] **Material Parameter Calibration**:
  - *Aluminium (Videos 1–7)*: Calibrate for high reflectivity, long ribbons, and tubular coils.
  - *Copper (Videos 8–12)*: Calibrate for ductile spiral coils and golden hue segmentation.
  - *Mild Steel (Videos 13–28)*: Calibrate for thermal discoloration oxidation and segmented C-chips.
- [ ] **Class Balance Verification** — Audit frame counts across materials and cutting parameters ($N$ in RPM, $a_p$ in mm, $f$ in mm/rev).

---

## Phase 6 — Dataset Annotation & Deep Learning Training

- [ ] **YOLO / Mask R-CNN Annotation (Roboflow / CVAT)**:
  - Annotate bounding boxes and segmentation polygons on `train/`, `val/`, and `test/` partitions for ISO 3685 chip types (*Ribbon, Tubular, Spiral, C-Shaped, Elemental*).
  - Include the 71 `NO_TOOL_BACKGROUND` negative samples with zero annotations to prevent false positives.
- [ ] **Model Training & Real-Time Video Inference Engine**:
  - Train YOLOv8-seg / YOLOv11 for real-time chip detection, morphology classification, and thermal process health monitoring.
  - Benchmark and evaluate performance on the held-out Video-Wise test set.
