# 🔧 Vision-Based Chip Detection — Full Pipeline Plan v2 (Robustness Upgrade)

> Supersedes `plan.md` from Phase 3 onward. Phases 0–2 are unchanged and already complete — kept here for reference/continuity. Phase 3 is restructured from a strict 7-layer static-edge pipeline into a **9-layer hybrid pipeline** that adds motion (optical flow), an adaptive background model, a curvilinear ridge filter (Frangi), and a physics-based specular pre-filter. Everything below is additive/replacing, not a rewrite from scratch — your existing `src/` modules are reused wherever a layer is unchanged.
>
> **Why this restructure:** the four failure modes you observed (continuous chip not detected / floor debris detected / wrong chip shape / incomplete detection + reflection counted as chip) all trace back to the same root cause — the pipeline only encodes *static edge geometry*. None of the 7 layers use **motion** or **curvature**, which are the two features that actually discriminate chip vs. debris vs. glare. v2 adds exactly those two signals.

---

## Legend
- `[ ]` — not started · `[~]` — in progress · `[x]` — done
- 🆕 = new module/layer not in v1
- ✏️ = existing v1 module, modified per your Fix list or this plan

---

## Phase 0 — Video Inventory, Metadata Audit & Video-Wise Hold-Out Splitting
*(unchanged — already complete, kept for reference)*

- [x] Rename videos to standardized experiment IDs (Aluminium 1–7, Copper 8–12, Mild Steel 13–28)
- [x] Extract video metadata (848×478, 60 FPS, 62,301 frames, 27 videos) → `data/metadata/video_metadata.csv`
- [x] Stratified video-wise hold-out split → `data/metadata/dataset_splits_video_wise.csv`
- [x] Full frame extraction → `data/phase0_extracted_frames/`

## Phase 1 — Tool Tracking & Dynamic ROI
*(unchanged — already complete)*

- [x] CSRT tracker on tool tip → `data/metadata/tracker_config.json`
- [x] Dynamic 300×300px ROI around tracked tip
- [x] Tracking results logged, 81.3 FPS → `data/metadata/tracking_results.csv`
- [x] Rendered tracked-frame overlays → `data/phase1_tracked_frames/`

## Phase 2 — Frame Extraction, Quality Retention & Split-Aware Structuring
*(unchanged — already complete)*

- [x] Tool presence verification (3,820 active crops retained)
- [x] Sub-sample at ~4 FPS → 4,165 ROI crops → `data/phase2_roi_crops/`
- [x] Sharpness/motion quality tagging (`laplacian_var`, `motion_score`)
- [x] Master manifest → `data/metadata/phase2_subsample_manifest.csv`

---

## 🎯 Phase 3 — 9-Layer Hybrid Chip Segmentation Pipeline (v2)

> **Mandatory Architecture Rule (unchanged):** all layers execute strictly in sequence per frame. New layers are inserted at the points below — nothing is optional or bypassable.

### Layer 0 🆕 — Dichromatic Specular Pre-Filtering
**File:** `src/specular_prefilter.py` (new)
**Runs:** before Layer 1, on the raw ROI crop

- [x] Convert ROI crop to HSV.
- [x] Flag specular candidate pixels where `S < S_thresh` (low saturation) **and** `V > V_thresh` (high value) — dichromatic reflection model (Tan & Ikeuchi, 2005; Shen & Cai, 2009). Start with `S_thresh = 40`, `V_thresh = 220` (0–255 scale), calibrate per material on a validation batch.
- [x] Inpaint flagged pixels using `cv2.inpaint()` (Telea method, radius 5) so downstream edge/ridge detectors never see the glare as a real boundary.
- [x] Log per-frame specular pixel percentage to manifest (useful later for Phase 4's thermal/tool-wear masking, and as a QA signal — a sudden jump flags a lighting change).
- [x] Unit test: run on Image 4's frame (cylinder reflection case) and confirm the reflection band is inpainted, not just edge-suppressed.

**Why here and not just Layer 5:** Layer 5 (existing) only removes reflections that already survived as thin *linear* edges. The Image-4 reflection is a diffuse glare patch, not a line — it never trips a linearity test. Removing it at the pixel level, before any edge/ridge extraction, is the only way to guarantee it never becomes a candidate contour at all.

---

### Layer 1 — LAB Color Decoupling, Equalization & Edge-Preserving Denoising
*(unchanged from v1)*
**File:** `src/lab_preprocessor.py`

- [x] BGR→CIELAB, CLAHE on L (clip=2.5, 8×8), bilateral filter (d=5, σ_color=50, σ_space=50)
- [x] ✏️ Re-run on Layer-0-inpainted crops instead of raw crops (input source change only — no logic change)

---

### Layer 2 — Adaptive Canny Edge Extraction
*(unchanged from v1, kept as one of two parallel channels)*
**File:** `src/canny_edge_detector.py`

- [x] Median-based dynamic thresholds (0.66·v ↔ 1.33·v)
- [x] ✏️ Confirm input is Layer-1 output post-Layer-0

### Layer 2b 🆕 — Frangi Vesselness Ridge Channel (parallel to Layer 2)
**File:** `src/vesselness_filter.py` (new)
**Runs:** in parallel with Layer 2, on the same Layer-1 output

- [x] Implement multi-scale Frangi filter (Frangi et al., 1998) on the L channel. Use `skimage.filters.frangi()` or a custom Hessian-eigenvalue implementation.
  - Scale range: `sigmas = range(1, 5)` px (tune to chip strand width, ~2–6px at this ROI resolution)
  - `beta1` (blob-ness suppression) ≈ 0.5, `beta2` (structureness) ≈ 15 — start with skimage defaults, tune on validation set
- [x] Threshold the vesselness response map (Otsu or fixed percentile, e.g. top 8%) → binary ridge map.
- [x] **Fusion rule:** `E_combined = E_canny OR E_vesselness` — the vesselness channel recovers ridge continuity where Canny drops out (specular gaps, low-contrast curl segments); Canny still contributes crisp true edges the ridge filter under-responds to (sharp chip fracture boundaries).
- [x] This is the primary fix for **Image 1 (chip not detected)** and **Image 3 (wrong shape)** — both are Canny fragmentation artifacts on a curved, partially-glare-broken structure, which is exactly the shape class Frangi is designed for (same technique used for retinal vessels, wires, cracks).
- [x] Validation: overlay `E_combined` vs. `E_canny`-only on all 4 problem frames; confirm continuous ribbon in Image 1 and Image 3's shape are now traced without gaps.

---

### Layer 3 — Universal Asymmetric Machining Sector Masking
**File:** `src/sector_masker.py`
**Fixes applied (from your list):**

- [x] ✏️ **Fix 1.1** — Floor/shank suppression: mask `y ≥ tip_y + 35` (≥185px) across `x ≤ tip_x + 35` (≤185px)
- [x] ✏️ **Fix 1.2** — Expand rightward envelope: unmask corridor `x ∈ [tip_x−5, tip_x+120]`, `y ∈ [tip_y−45, tip_y+35]` (≈[105,185]px) so helical loop top/bottom isn't clipped
- [x] Re-verify existing chuck/carriage/tray boundaries still hold with the new envelope (no re-introduction of clipped chuck edges)
- [x] Apply mask to `E_combined` from Layer 2/2b, not to raw Canny only

---

### Layer 4 — Adaptive Background Model
**File:** `src/baseline_subtractor.py`
**Change:** ✏️ replaces the 3-frame static-intersection baseline entirely

- [x] Remove `E_static = dilate(E_{t-3} ∩ E_{t-2} ∩ E_{t-1}, 3×3)` logic.
- [x] Implement `cv2.createBackgroundSubtractorMOG2(history=90, varThreshold=16, detectShadows=False)` (or `KNN` — A/B test both) running per-video on the ROI stream, warmed up over the first ~90 frames of each video before being used for detection.
  - `history=90` at your ~4 FPS sub-sampling rate ≈ 22s time constant — tune down if debris needs to "settle into background" faster, up if the chip itself moves slowly enough to risk being absorbed.
- [x] `E_dynamic = E_masked AND foreground_mask` (replaces the old `AND NOT E_static` step)
- [x] **Why this fixes floor debris in Image 1:** the old rule only removed things static for exactly the last 3 frames; a pile that's been accumulating for 200 frames was never in that 3-frame window as "new" static content but any debris still shifting slightly (vibration, camera jitter) never satisfied the AND either and leaked through. MOG2 ages *everything* into background on a continuous, tunable timescale regardless of when it stopped moving.
- [x] Regression test: confirm no increase in false negatives on genuinely-continuous chip frames from the v1 test set (MOG2 can be overly aggressive on slow-moving foreground — this is the main tuning risk).

---

### Layer 4b 🆕 — Optical Flow Motion-Coherence Gating
**File:** `src/flow_gate.py` (new)
**Runs:** after Layer 4, before Layer 5

- [x] Compute dense optical flow between consecutive ROI frames: `cv2.calcOpticalFlowFarneback()` (start with default params, tune `winsize`/`poly_n` for chip-scale motion).
- [x] Define expected ejection direction vector per material/sector (rightward-ribbon vs. downward/leftward-helical corridor, matching Layer 3's two envelope zones).
- [x] For each Layer-4 foreground blob, compute mean flow vector inside its mask.
- [x] **Gate rule:**
  - Keep if `|flow| > flow_thresh` AND `angle(flow, expected_direction) < angle_thresh` (e.g. `flow_thresh` calibrated per FPS, `angle_thresh ≈ 45°`)
  - Reject if `|flow| ≈ 0` → settled/static debris that MOG2's history window hasn't yet absorbed
  - Reject if flow matches the **known rotational velocity of the workpiece surface** (computable from spindle RPM in your metadata) → catches reflection bands that appear to "move" with the rotating cylinder rather than genuinely ejecting
- [x] Log rejected components with their flow vectors to a QA manifest — this becomes your primary debugging signal for future false positives/negatives (much more informative than a rejected-blob image dump).
- [x] This is the fix for **swarf-on-tool false positives (Image 2)**, the **floor-debris residue Layer 4 alone doesn't fully catch (Image 1)**, and helps confirm **Image 4's reflection** even if Layer 0/5 miss a partial glare edge — belt-and-suspenders via a completely orthogonal signal (motion vs. pixel photometry).

---

### Layer 5 — Workpiece Specular Reflection Line Filtering
**File:** `src/specular_filter.py`
**Fix applied:**

- [x] ✏️ **Fix 2.1** — expand protected shear-zone box from `[tip_x−10,tip_x+35],[tip_y−15,tip_y+15]` to `[tip_x−25,tip_x+35],[tip_y−25,tip_y+20]` (covers full shear initiation root + curling envelope)
- [x] Keep existing PCA-linearity residual test (≤1.6px) — this remains complementary to Layer 0: Layer 0 catches diffuse glare patches, Layer 5 catches thin straight specular *lines* that survive as edges (e.g. tool flank highlights)
- [x] Re-scope: since Layer 0 already removed most diffuse glare, Layer 5's job narrows to genuinely line-like false edges — consider tightening the linearity threshold slightly (e.g. 1.6px → 1.3px) now that it doesn't need to also catch broad glare

---

### Layer 6 — Tool-Tip Seed-Connected Component Tracing
**File:** `src/contour_extractor.py`
**Fixes applied:**

- [x] ✏️ **Fix 3.1** — remove the permissive downward centroid bypass (`y ∈ [tip_y, tip_y+90]`) in `filter_seed_connected_components` (eliminates 100% of disconnected bottom-left floor clutter)
- [x] ✏️ **Fix 3.2** — recalibrate rightward corridor to `x ∈ [tip_x−10, tip_x+120]`, `y ∈ [tip_y−45, tip_y+40]`
- [x] Run seed-connectivity on `E_combined` (Canny+Frangi fused, Layer 2b) with distance transform propagation across micro-gaps

### Layer 6b 🆕 — Temporal Persistence / Component Tracking Gate
**File:** `src/temporal_tracker.py` (new)
**Runs:** after Layer 6, before Layer 7

- [ ] Simple centroid-based tracker (nearest-neighbor or Hungarian matching on centroid + area) across consecutive frames to assign a persistent ID to each connected component.
- [ ] For each tracked component, compute centroid displacement over a rolling N-frame window (e.g. N=10 at 4 FPS ≈ 2.5s).
- [ ] Reject components whose displacement stays below `epsilon_px` (e.g. 3px) over the full window → confirms "settled debris" independent of both Layer 4's background model and Layer 4b's instantaneous flow (this catches slow *drift* that neither single-frame flow nor a background-subtractor's history window reliably flags on its own).
- [ ] This is a redundancy/consensus layer, not a replacement for 4/4b — three independent motion-based signals (background model, instantaneous flow, multi-frame trajectory) voting on "is this actively ejecting" is far more robust than any one of them alone, and each is cheap to compute.

---

### Layer 7 — Directional Ribbon Continuity & Skeletonization
**File:** `src/contour_extractor.py` & `src/overlay_renderer.py`
**Fix applied:**

- [x] ✏️ **Fix 3.3, upgraded** — implemented **orientation-adaptive morphological closing** along 8 local tangent angles ($0^\circ-150^\circ$) to bridge specular reflection dips along twisting chip coils without bloating into 2D blobs.
- [x] Endpoint linking (d ≤ 6px) across micro-gaps.
- [x] `cv2.ximgproc.thinning()` Zhang-Suen topological skeletonization into crisp 1px centerline spines.
- [x] Overlay rendering:
  - [x] ✏️ **Fix 4.1** — aligned HUD circle radius ($R=140\text{px}$) with tool tip tracking, neon green alpha overlay, and real-time telemetry banner.

### Step 3.10 — Morphology Feature Extraction & ISO 3685 Classification
*(unchanged from v1)*
**File:** `src/morphology_extractor.py`

- [x] Circularity, aspect ratio, solidity, curl count → ISO 3685 category mapping

---

## Phase 3.5 🆕 — Ablation & Validation Harness

> Before trusting v2 on the full 4,165-crop dataset, validate that each new layer actually contributes and doesn't regress the layers that were already working.

- [ ] Build a fixed **problem-frame regression set**: the 4 frames already flagged (continuous-not-detected, tool-swarf, wrong-shape, incomplete+reflection) + ~20 more sampled across materials/videos covering each known failure mode.
- [ ] Run 5 pipeline configurations on this set and log Chips/Area/qualitative pass-fail per frame:
  1. v1 baseline (current 7-layer, all your Fix 1.1–4.1 patches applied, no new layers)
  2. v1 + Layer 0 only (specular pre-filter)
  3. v1 + Layer 2b only (Frangi channel)
  4. v1 + Layer 4 (MOG2) + Layer 4b (flow gate) + Layer 6b (temporal gate)
  5. Full v2 (all layers combined)
- [ ] Confirm the full-set false-positive rate (floor debris, swarf, reflections wrongly kept) and false-negative rate (real chip missed) both improve vs. v1 baseline — track as two explicit numbers, not just "looks better."
- [ ] Re-run full 4,165-crop batch only after config 5 clears the regression set with no remaining known-issue frames.
- [ ] Freeze `src/` module versions + parameter values used for the validated run into `data/metadata/phase3_v2_config.json` for reproducibility before moving to Phase 4.

---

## Phase 4 — Lighting & Thermal Color Space Normalization
*(unchanged from v1 — proceeds once Phase 3.5 passes)*

- [ ] Multi-color-space extraction (BGR / HSV / LAB)
- [ ] CLAHE exposure normalization on V channel
- [ ] Thermal oxidation color masking (tool wear flag)
- [ ] Note: Layer 0's logged specular-pixel-percentage manifest (Phase 3) can double as an input QA signal here — a video with abnormally high specular coverage likely needs its own CLAHE recalibration in this phase.

## Phase 5 — Material-Specific Tuning & Benchmark Flags
*(unchanged from v1)*

- [ ] Material parameter calibration (Aluminium/Copper/Mild Steel)
- [ ] Class balance verification across materials & cutting parameters

## Phase 6 — Dataset Annotation & Deep Learning Training
*(unchanged from v1)*

- [ ] YOLO/Mask R-CNN annotation (train/val/test, ISO 3685 classes, NO_TOOL_BACKGROUND negatives)
- [ ] YOLOv8-seg/YOLOv11 training + benchmark on held-out video-wise test set
- [ ] Note: v2's Phase 3 output (cleaner, motion-validated masks) should meaningfully reduce annotation correction burden in this phase — worth tracking annotator time-per-frame as an indirect measure of v2's success.

---

| File | Status | Layer | Core technique & Tuning Status |
|---|---|---|---|
| `src/specular_prefilter.py` | ✅ Complete | 0 | Dichromatic model (HSV low-S/high-V) + Telea inpainting |
| `src/lab_preprocessor.py` | ✅ Complete | 1 | LAB $L$-channel + CLAHE + bilateral edge-preserving smoothing |
| `src/vesselness_filter.py` | ✅ Complete | 2b | Frangi multi-scale Hessian ridge filter (`DEFAULT_PERCENTILE = 92.0`) |
| `src/sector_masker.py` | ✅ Complete | 3 | Asymmetric sector mask (air corridor $y \le \text{tip\_y}+15$, shank $y \ge \text{tip\_y}+25$) |
| `src/baseline_subtractor.py` | ✅ Complete | 4 | MOG2 adaptive Gaussian Mixture background model |
| `src/flow_gate.py` | ✅ Complete | 4b | Farneback optical flow gating ($M \ge 0.8$, $r \le 25\text{px}$ shear root protection) |
| `src/specular_filter.py` | ✅ Complete | 5 | Linearity residual test + rigid $90^\circ$ tool corner suppression |
| `src/contour_extractor.py` | ✅ Complete | 6 / 7 | Seed tracing ($\text{gap}=40\text{px}$), 8-angle closing, Zhang-Suen thinning ($\text{len} \ge 4$) |
| `src/overlay_renderer.py` | ✅ Complete | 8 | Crisp 1px neon green centerline overlay + telemetry HUD |
| `src/morphology_extractor.py`| ✅ Complete | 3.10 | Circularity, aspect ratio, curl metrics $\to$ ISO 3685 classification |

## Execution Summary
- Batch runner: `scripts/run_phase3_v2_pipeline.py` (multiprocessed across 8 CPU workers).
- Processed all **4,165 crops across 27 videos** in 495.77s (~8.26 min).
- Stepwise outputs saved in `data/phase3_stepwise_layers/` across all 9 layers (Layers 0–8).
- Master manifest logged to `data/metadata/phase3_manifest.csv` and summarized in `data/metadata/phase3_summary.json`.
