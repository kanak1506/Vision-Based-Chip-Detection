# Vision-Based Chip Detection — v2 Pipeline: Summary

The full v2 plan (`planv2.md`) supersedes Phase 3 of the original `plan.md`. Phases 0–2 are kept as-is for continuity.

## Structure of the Active 9-Layer Pipeline

- **Layer 0 ✅** — Dichromatic specular pre-filtering (`src/specular_prefilter.py`): Inpaints HSV low-S/high-V glare before edge detection to eliminate diffuse cylinder reflections.
- **Layer 1 ✅** — LAB decoupling, CLAHE ($8\times8$, clip=2.5), and bilateral edge-preserving filter (`src/lab_preprocessor.py`).
- **Layer 2 / 2b ✅** — Multi-scale Frangi curvilinear ridge filter (`src/vesselness_filter.py`) with 92.0% threshold (top 8% response) to detect continuous ribbons, helical curls, and elemental chips.
- **Layer 3 ✅** — Asymmetric sector masking (`src/sector_masker.py`): Air-zone rightward ribbon envelope ($y \in [\text{tip\_y}-60, \text{tip\_y}+15]$), downward helical corridor, and carriage/tool-shank suppression ($y \ge \text{tip\_y}+25$).
- **Layer 4 ✅** — Adaptive background model (`src/baseline_subtractor.py`): MOG2 Gaussian Mixture Model for dynamic foreground extraction.
- **Layer 4b ✅** — Farneback dense optical flow gating (`src/flow_gate.py`): Rejects stationary debris and tool swarf ($M \ge 0.8\text{ px/frame}$) with immediate $25\text{px}$ shear root protection.
- **Layer 5 ✅** — Specular line & rigid tool corner filter (`src/specular_filter.py`): Suppresses straight reflection lines ($R^2 > 0.92, |\theta| \le 16^\circ$) and $90^\circ$ tool post corners.
- **Layer 6 ✅** — Seed-connected component tracing (`src/contour_extractor.py`): Multi-hop geodesic chaining originating at tool tip ($r=25\text{px}$) with $\text{max\_gap\_ribbon} = 40.0\text{px}$.
- **Layer 7 ✅** — Orientation-adaptive morphological closing (8 tangent directions), filled medial axis, and Zhang-Suen 1px skeletonization (`min_contour_length = 4\text{px}`).
- **Layer 8 ✅** — Real-time telemetry HUD & crisp single-line neon green contour overlay rendering (`src/overlay_renderer.py`).
- **Step 3.10 ✅** — Morphology feature extraction to ISO 3685 standard classes (`src/morphology_extractor.py`).

## Implemented Module Summary

| File | Status | Layer | Core Technique & Recent Tunings |
|---|---|---|---|
| `src/specular_prefilter.py` | ✅ Implemented | 0 | Dichromatic reflection model (HSV low-S/high-V) + Telea inpainting |
| `src/lab_preprocessor.py` | ✅ Implemented | 1 | CIELAB Luminance extraction + CLAHE + bilateral filtering |
| `src/vesselness_filter.py` | ✅ Implemented | 2b | Multi-scale Hessian Frangi filter (`DEFAULT_PERCENTILE = 92.0`) |
| `src/sector_masker.py` | ✅ Implemented | 3 | Asymmetric sector mask (air corridor $y \le \text{tip\_y}+15$, shank $y \ge \text{tip\_y}+25$) |
| `src/baseline_subtractor.py` | ✅ Implemented | 4 | MOG2 Gaussian Mixture background subtractor |
| `src/flow_gate.py` | ✅ Implemented | 4b | Dense optical flow gating ($M \ge 0.8$, $r \le 25\text{px}$ shear root protection) |
| `src/specular_filter.py` | ✅ Implemented | 5 | Linearity residual filter + approxPolyDP rigid $90^\circ$ tool corner suppression |
| `src/contour_extractor.py` | ✅ Implemented | 6 / 7 | Seed tracing ($\text{gap}=40\text{px}$), 8-angle closing, Zhang-Suen thinning ($\text{len} \ge 4$) |
| `src/overlay_renderer.py` | ✅ Implemented | 8 | Crisp 1px neon green centerline overlay + telemetry banner |
| `src/morphology_extractor.py`| ✅ Implemented | 3.10 | Circularity, aspect ratio, curl metrics $\to$ ISO 3685 classification |

## Execution & Verification Status
- Full batch execution completed across all **4,165 crops (27 videos)** in `data/phase3_stepwise_layers/`.
- Manifest cataloged in `data/metadata/phase3_manifest.csv` and summarized in `data/metadata/phase3_summary.json`.
