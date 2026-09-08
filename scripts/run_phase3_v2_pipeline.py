# -*- coding: utf-8 -*-
"""
Phase 3 v2 Hybrid Chip Segmentation Pipeline Runner (planv2.md)
================================================================
Complete 9-layer sequential pipeline from planv2.md.

Fully Implemented & Active Layers:
  - Layer 0:  Dichromatic Specular Pre-Filtering (src/specular_prefilter.py)
  - Layer 1:  LAB Decoupling, CLAHE Equalization & Bilateral Denoising (src/lab_preprocessor.py)
  - Layer 2a: Adaptive Dynamic Canny Edge Extraction (src/canny_edge_detector.py)
  - Layer 2b: Multi-Scale Frangi Curvilinear Ridge Channel (src/vesselness_filter.py)
  - Layer 2c: Fused Representation (E_combined = E_canny OR E_vesselness)
  - Layer 3:  Universal Asymmetric Machining Sector Masking with Expanded Helical Envelope (src/sector_masker.py)
  - Layer 4:  Adaptive Background Model (MOG2) (src/baseline_subtractor.py)
  - Layer 4b: Dense Optical Flow Motion-Coherence Gating (src/flow_gate.py)
  - Layer 5:  Workpiece Specular Line & Corner Filtering (src/specular_filter.py, Fix 2.1 & 2.2)
  - Layer 6:  Seed-Connected Component Tracing (src/contour_extractor.py, Fix 3.1 & 3.2)
  - Layer 7:  Orientation-Adaptive Morphological Closing & 1px Skeletonization (src/contour_extractor.py, Fix 3.3)
  - Layer 8:  Real-Time HUD Telemetry & Chip Segmentation Overlays (src/overlay_renderer.py)
"""

import os
import sys
import time
import json
import cv2
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    PROJECT_ROOT,
    PHASE2_CROPS_DIR,
    PHASE3_STEPWISE_DIR,
    PHASE3_L0_SPECULAR_DIR,
    PHASE3_L1_LAB_DIR,
    PHASE3_L2_FRANGI_DIR,
    PHASE3_L3_SECTOR_DIR,
    PHASE3_L4_ADAPTIVE_BG_DIR,
    PHASE3_L4B_FLOW_DIR,
    PHASE3_L5_SPECULAR_DIR,
    PHASE3_L6_SEED_CONNECTED_DIR,
    PHASE3_L7_SKELETON_DIR,
    PHASE3_L8_OVERLAYS_DIR,
    PHASE2_MANIFEST_PATH,
    PHASE3_MANIFEST_PATH,
    to_relative_path,
    N_WORKERS
)
from src.specular_prefilter import apply_specular_prefilter
from src.lab_preprocessor import preprocess_lightness_pipeline
from src.vesselness_filter import apply_vesselness_filter
from src.sector_masker import apply_sector_mask
from src.baseline_subtractor import AdaptiveBackgroundModel
from src.flow_gate import OpticalFlowGate
from src.specular_filter import filter_specular_lines
from src.contour_extractor import filter_seed_connected_components, extract_chip_contours
from src.overlay_renderer import render_chip_overlay

V2_SUMMARY_PATH = os.path.join(os.path.dirname(PHASE3_MANIFEST_PATH), "phase3_summary.json")


def process_video_v2(task: tuple) -> dict:
    (
        v_clean_name, records,
        l0_base, l1_base, l2_base, l3_base, l4_base, l4b_base, l5_base, l6_base, l7_base, l8_base
    ) = task

    l0_out_dir = os.path.join(l0_base, "retained", v_clean_name)
    l1_out_dir = os.path.join(l1_base, "retained", v_clean_name)
    l2_out_dir = os.path.join(l2_base, "retained", v_clean_name)
    l3_out_dir = os.path.join(l3_base, "retained", v_clean_name)
    l4_out_dir = os.path.join(l4_base, "retained", v_clean_name)
    l4b_out_dir = os.path.join(l4b_base, "retained", v_clean_name)
    l5_out_dir = os.path.join(l5_base, "retained", v_clean_name)
    l6_out_dir = os.path.join(l6_base, "retained", v_clean_name)
    l7_out_dir = os.path.join(l7_base, "retained", v_clean_name)
    l8_out_dir = os.path.join(l8_base, "retained", v_clean_name)

    for d in (l0_out_dir, l1_out_dir, l2_out_dir, l3_out_dir, l4_out_dir, l4b_out_dir, l5_out_dir, l6_out_dir, l7_out_dir, l8_out_dir):
        os.makedirs(d, exist_ok=True)

    sorted_records = sorted(records, key=lambda r: int(r["frame_idx"]))
    updated_records = []

    # Initialize per-video Adaptive Background Subtractor & Optical Flow Gate
    bg_model = AdaptiveBackgroundModel(method="MOG2", history=90, var_threshold=16.0, detect_shadows=False)
    flow_gate = OpticalFlowGate(min_magnitude=1.2, tip_protect_radius=25, mask_dilation_ksize=5)

    for row in sorted_records:
        crop_path = row["crop_path"]
        abs_crop_path = os.path.join(PROJECT_ROOT, crop_path) if not os.path.isabs(crop_path) else crop_path

        if not os.path.exists(abs_crop_path):
            continue

        img_bgr = cv2.imread(abs_crop_path)
        if img_bgr is None:
            continue
        if img_bgr.shape[:2] != (300, 300):
            img_bgr = cv2.resize(img_bgr, (300, 300))

        f_idx = int(row["frame_idx"])
        tip_x = int(row.get("crop_tip_x", 150))
        tip_y = int(row.get("crop_tip_y", 150))

        # Layer 0: Dichromatic Specular Pre-Filter
        img_prefiltered, specular_mask, specular_pct = apply_specular_prefilter(img_bgr)

        # Layer 1: LAB Decoupling + CLAHE + Bilateral Denoising
        l_denoised = preprocess_lightness_pipeline(img_prefiltered)

        # Layer 2: Multi-Scale Frangi Curvilinear Ridge Filter (Pure Frangi Vesselness)
        vmap, ridge_binary = apply_vesselness_filter(l_denoised)

        # Layer 3: Universal Asymmetric Machining Sector Masking
        masked_edges, sector_mask = apply_sector_mask(ridge_binary, tip_x=tip_x, tip_y=tip_y, cutting_radius=135)

        # Layer 4: Adaptive Background Subtraction (MOG2)
        dynamic_edges, fg_mask = bg_model.filter_edges(masked_edges, img_prefiltered)

        # Layer 4b: Optical Flow Motion-Coherence Gating
        gated_edges, motion_mask, magnitude = flow_gate.apply(img_prefiltered, dynamic_edges, tip_x=tip_x, tip_y=tip_y)

        # Layer 5: Workpiece Specular Line & Corner Filtering (Fix 2.1 & Fix 2.2)
        clean_chip_edges, removed_specular = filter_specular_lines(gated_edges, tip_x=tip_x, tip_y=tip_y)

        # Layer 6: Seed-Connected Component Tracing (Fix 3.1 & Fix 3.2)
        seed_connected_edges, discarded_clutter = filter_seed_connected_components(clean_chip_edges, tip_x=tip_x, tip_y=tip_y)

        # Layer 7: Orientation-Adaptive Closing, Skeletonization & Metric Extraction
        skeleton_map, contours, metrics = extract_chip_contours(seed_connected_edges, tip_x=tip_x, tip_y=tip_y, max_dist_to_tip=140.0)

        # Layer 8: Chip HUD Overlay Rendering (Clean: no circles, no dots)
        overlay_img = render_chip_overlay(img_bgr, skeleton_map, tip_x=tip_x, tip_y=tip_y, proximity_radius=0, metrics=metrics)

        # Save Stepwise Outputs
        crop_png_name = f"crop_{f_idx:06d}.png"
        cv2.imwrite(os.path.join(l0_out_dir, crop_png_name), img_prefiltered)
        cv2.imwrite(os.path.join(l1_out_dir, crop_png_name), l_denoised)
        cv2.imwrite(os.path.join(l2_out_dir, crop_png_name), ridge_binary)
        cv2.imwrite(os.path.join(l3_out_dir, crop_png_name), masked_edges)
        cv2.imwrite(os.path.join(l4_out_dir, crop_png_name), dynamic_edges)
        cv2.imwrite(os.path.join(l4b_out_dir, crop_png_name), gated_edges)
        cv2.imwrite(os.path.join(l5_out_dir, crop_png_name), clean_chip_edges)
        cv2.imwrite(os.path.join(l6_out_dir, crop_png_name), seed_connected_edges)
        cv2.imwrite(os.path.join(l7_out_dir, crop_png_name), skeleton_map)
        cv2.imwrite(os.path.join(l8_out_dir, crop_png_name), overlay_img)

        record = dict(row)
        record["layer0_prefiltered_path"] = to_relative_path(os.path.join(l0_out_dir, crop_png_name))
        record["layer1_denoised_path"] = to_relative_path(os.path.join(l1_out_dir, crop_png_name))
        record["layer2_frangi_vesselness_path"] = to_relative_path(os.path.join(l2_out_dir, crop_png_name))
        record["layer3_sector_masked_path"] = to_relative_path(os.path.join(l3_out_dir, crop_png_name))
        record["layer4_adaptive_bg_path"] = to_relative_path(os.path.join(l4_out_dir, crop_png_name))
        record["layer4b_flow_gated_path"] = to_relative_path(os.path.join(l4b_out_dir, crop_png_name))
        record["layer5_specular_filtered_path"] = to_relative_path(os.path.join(l5_out_dir, crop_png_name))
        record["layer6_seed_connected_path"] = to_relative_path(os.path.join(l6_out_dir, crop_png_name))
        record["layer7_skeleton_path"] = to_relative_path(os.path.join(l7_out_dir, crop_png_name))
        record["layer8_overlay_path"] = to_relative_path(os.path.join(l8_out_dir, crop_png_name))
        record["specular_pct"] = round(specular_pct, 4)
        record["num_chip_clusters"] = metrics["num_chip_clusters"]
        record["total_chip_area"] = metrics["total_chip_area"]
        record["total_chip_perimeter"] = metrics["total_chip_perimeter"]
        record["min_dist_to_tip"] = metrics["min_dist_to_tip"]
        updated_records.append(record)

    return {"clean_name": v_clean_name, "processed_count": len(updated_records), "records": updated_records}


def run_phase3_v2_batch(workers: int = N_WORKERS):
    start_time = time.time()
    df_phase2 = pd.read_csv(PHASE2_MANIFEST_PATH)
    df_retained = df_phase2[df_phase2["category"] == "RETAINED"].copy()

    tasks = []
    for _, group in df_retained.groupby("video"):
        clean_name = os.path.basename(os.path.dirname(group.iloc[0]["crop_path"]))
        records = group.to_dict("records")
        tasks.append((
            clean_name, records,
            PHASE3_L0_SPECULAR_DIR,
            PHASE3_L1_LAB_DIR,
            PHASE3_L2_FRANGI_DIR,
            PHASE3_L3_SECTOR_DIR,
            PHASE3_L4_ADAPTIVE_BG_DIR,
            PHASE3_L4B_FLOW_DIR,
            PHASE3_L5_SPECULAR_DIR,
            PHASE3_L6_SEED_CONNECTED_DIR,
            PHASE3_L7_SKELETON_DIR,
            PHASE3_L8_OVERLAYS_DIR
        ))

    print("=" * 75)
    print(f"  PHASE 3 v2: COMPLETE PIPELINE BATCH EXECUTION")
    print(f"  Total Retained Crops: {len(df_retained):,}")
    print(f"  Total Videos:         {len(tasks)}")
    print(f"  Stepwise Root:        '{PHASE3_STEPWISE_DIR}'")
    print(f"  Active Layers:        Layer 0 to Layer 8 (Pure Frangi Vesselness Stack)")
    print(f"  Parallel CPU Workers: {workers}")
    print("=" * 75)

    all_records = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_video_v2, t): t[0] for t in tasks}
        for future in as_completed(futures):
            c_name = futures[future]
            res = future.result()
            all_records.extend(res["records"])
            print(f"  [v2] Processed {c_name} ({res['processed_count']} frames)")

    elapsed = time.time() - start_time
    df_v2 = pd.DataFrame(all_records)
    os.makedirs(os.path.dirname(PHASE3_MANIFEST_PATH), exist_ok=True)
    df_v2.to_csv(PHASE3_MANIFEST_PATH, index=False)

    summary = {
        "pipeline_version": "v2_hybrid_9layer",
        "total_crops_processed": len(df_v2),
        "total_videos": len(tasks),
        "elapsed_seconds": round(elapsed, 2),
        "stepwise_layers_root": to_relative_path(PHASE3_STEPWISE_DIR),
        "active_verified_layers": [
            "layer0_specular_prefiltered",
            "layer1_lab_denoised",
            "layer2_frangi_vesselness",
            "layer3_sector_masked",
            "layer4_adaptive_background",
            "layer4b_optical_flow_gated",
            "layer5_specular_filtered",
            "layer6_seed_connected_chips",
            "layer7_morphological_skeleton",
            "layer8_chip_overlays"
        ],
        "manifest_path": to_relative_path(PHASE3_MANIFEST_PATH)
    }
    with open(V2_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 75)
    print(f"  Phase 3 v2 Complete 9-Layer Batch Processing Complete in {elapsed:.2f}s (~{elapsed/60:.2f} min)")
    print(f"  Saved Layer 0 (Specular Inpainted) to: '{PHASE3_L0_SPECULAR_DIR}'")
    print(f"  Saved Layer 1 (LAB/CLAHE/Bilateral) to: '{PHASE3_L1_LAB_DIR}'")
    print(f"  Saved Layer 2 (Frangi Vesselness) to:  '{PHASE3_L2_FRANGI_DIR}'")
    print(f"  Saved Layer 3 (Sector Masked) to:      '{PHASE3_L3_SECTOR_DIR}'")
    print(f"  Saved Layer 4 (Adaptive MOG2 BG) to:   '{PHASE3_L4_ADAPTIVE_BG_DIR}'")
    print(f"  Saved Layer 4b (Optical Flow Gated) to:'{PHASE3_L4B_FLOW_DIR}'")
    print(f"  Saved Layer 5 (Specular Filtered) to:  '{PHASE3_L5_SPECULAR_DIR}'")
    print(f"  Saved Layer 6 (Seed Connected) to:     '{PHASE3_L6_SEED_CONNECTED_DIR}'")
    print(f"  Saved Layer 7 (1px Skeleton) to:       '{PHASE3_L7_SKELETON_DIR}'")
    print(f"  Saved Layer 8 (HUD Overlays) to:       '{PHASE3_L8_OVERLAYS_DIR}'")
    print(f"  Saved Phase 3 Manifest to:             '{PHASE3_MANIFEST_PATH}'")
    print(f"  Saved Phase 3 Summary to:              '{V2_SUMMARY_PATH}'")
    print("=" * 75)


if __name__ == "__main__":
    run_phase3_v2_batch()
