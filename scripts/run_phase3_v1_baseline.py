# -*- coding: utf-8 -*-
"""
Phase 3 v1 Baseline Pipeline Runner (plan.md)
==============================================
Pure static 7-layer pipeline implementation from plan.md (v1 baseline):
  - Layer 1-4: Raw Crop -> LAB -> CLAHE -> Bilateral -> Adaptive Canny
  - Layer 5: Asymmetric Cutting Sector Masking (v1 boundaries)
  - Layer 6: 3-Frame Temporal Static Intersection Baseline Subtraction
  - Layer 7: Workpiece Specular Line Filtering (v1 parameters)
  - Layer 8: Tool-Tip Seed-Connected Component Tracing (v1 parameters)
  - Layer 9: Ribbon Continuity & Skeletonization
  - Step 3.10: Morphology Metrics & ISO 3685 Classification

Preserved strictly for Phase 3.5 ablation comparison against planv2.md.
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
    DATA_DIR,
    PHASE2_CROPS_DIR,
    PHASE2_MANIFEST_PATH,
    METADATA_DIR,
    to_relative_path,
    N_WORKERS
)
from src.canny_edge_detector import extract_chip_edges_pipeline
from src.sector_masker import apply_sector_mask
from src.baseline_subtractor import compute_dilated_baseline, compute_temporal_static_baseline, subtract_baseline_edges
from src.specular_filter import filter_specular_lines
from src.contour_extractor import extract_chip_contours, filter_seed_connected_components
from src.overlay_renderer import render_chip_overlay
from src.morphology_extractor import compute_morphology_metrics

V1_OUTPUT_DIR = DATA_DIR / "phase3_v1_baseline"
V1_MANIFEST_PATH = METADATA_DIR / "phase3_v1_manifest.csv"
V1_SUMMARY_PATH = METADATA_DIR / "phase3_v1_summary.json"


def process_video_v1(task: tuple) -> dict:
    v_clean_name, records, out_base = task

    canny_out = os.path.join(out_base, "step1_canny", "retained", v_clean_name)
    masked_out = os.path.join(out_base, "step2_masked", "retained", v_clean_name)
    sub_out = os.path.join(out_base, "step3_subtracted", "retained", v_clean_name)
    spec_out = os.path.join(out_base, "step4_specular", "retained", v_clean_name)
    cnt_out = os.path.join(out_base, "step5_contours", "retained", v_clean_name)
    ovl_out = os.path.join(out_base, "step6_overlays", "retained", v_clean_name)

    for d in (canny_out, masked_out, sub_out, spec_out, cnt_out, ovl_out):
        os.makedirs(d, exist_ok=True)

    sorted_records = sorted(records, key=lambda r: int(r["frame_idx"]))
    edge_history = []
    updated_records = []

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

        # v1 Step 1-4: Raw crop -> LAB -> CLAHE -> Bilateral -> Canny
        _, canny_edges = extract_chip_edges_pipeline(img_bgr)

        # v1 Step 3.5: Asymmetric Sector Masking
        masked_edges, _ = apply_sector_mask(canny_edges, tip_x=tip_x, tip_y=tip_y, cutting_radius=135)

        # v1 Step 3.6: 3-Frame Static Intersection Baseline Subtraction
        temporal_static = compute_temporal_static_baseline(edge_history, current_shape=img_bgr.shape, kernel_size=3)
        edge_history.append(masked_edges)
        if len(edge_history) > 5:
            edge_history.pop(0)
        subtracted_edges = subtract_baseline_edges(masked_edges, temporal_static)

        # v1 Step 3.7: Specular Line Filtering
        clean_chip_edges, _ = filter_specular_lines(subtracted_edges, tip_x=tip_x, tip_y=tip_y, max_linearity_residual=1.6, min_line_len=10)

        # v1 Step 3.8: Seed-Connected Components
        seed_connected = filter_seed_connected_components(clean_chip_edges, tip_x=tip_x, tip_y=tip_y)

        # v1 Step 3.8-3.9: Contours & Overlays
        chip_mask, contours, metrics = extract_chip_contours(seed_connected, tip_x=tip_x, tip_y=tip_y, max_dist_to_tip=120.0, min_contour_length=10, max_link_dist=5.0)
        overlay_img = render_chip_overlay(img_bgr, chip_mask, tip_x=tip_x, tip_y=tip_y, proximity_radius=125, alpha=0.0, metrics=metrics)
        morph = compute_morphology_metrics(contours, shape=img_bgr.shape)

        png_name = f"crop_{f_idx:06d}.png"
        jpg_name = f"crop_{f_idx:06d}.jpg"

        cv2.imwrite(os.path.join(canny_out, png_name), canny_edges)
        cv2.imwrite(os.path.join(masked_out, png_name), masked_edges)
        cv2.imwrite(os.path.join(sub_out, png_name), subtracted_edges)
        cv2.imwrite(os.path.join(spec_out, png_name), clean_chip_edges)
        cv2.imwrite(os.path.join(cnt_out, png_name), chip_mask)
        cv2.imwrite(os.path.join(ovl_out, jpg_name), overlay_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        record = dict(row)
        record["v1_canny_path"] = to_relative_path(os.path.join(canny_out, png_name))
        record["v1_masked_path"] = to_relative_path(os.path.join(masked_out, png_name))
        record["v1_subtracted_path"] = to_relative_path(os.path.join(sub_out, png_name))
        record["v1_specular_path"] = to_relative_path(os.path.join(spec_out, png_name))
        record["v1_contour_path"] = to_relative_path(os.path.join(cnt_out, png_name))
        record["v1_overlay_path"] = to_relative_path(os.path.join(ovl_out, jpg_name))
        record["v1_num_chip_clusters"] = metrics["num_chip_clusters"]
        record["v1_total_chip_area"] = metrics["total_chip_area"]
        record["v1_iso_3685_class"] = morph["iso_3685_class"]
        updated_records.append(record)

    return {"clean_name": v_clean_name, "processed_count": len(updated_records), "records": updated_records}


def run_phase3_v1_batch(workers: int = N_WORKERS):
    start_time = time.time()
    df_phase2 = pd.read_csv(PHASE2_MANIFEST_PATH)
    df_retained = df_phase2[df_phase2["category"] == "RETAINED"].copy()

    tasks = []
    for _, group in df_retained.groupby("video"):
        clean_name = os.path.basename(os.path.dirname(group.iloc[0]["crop_path"]))
        records = group.to_dict("records")
        tasks.append((clean_name, records, V1_OUTPUT_DIR))

    all_records = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_video_v1, t): t[0] for t in tasks}
        for future in as_completed(futures):
            c_name = futures[future]
            res = future.result()
            all_records.extend(res["records"])
            print(f"  [v1] Processed {c_name} ({res['processed_count']} frames)")

    elapsed = time.time() - start_time
    df_v1 = pd.DataFrame(all_records)
    os.makedirs(os.path.dirname(V1_MANIFEST_PATH), exist_ok=True)
    df_v1.to_csv(V1_MANIFEST_PATH, index=False)

    summary = {
        "pipeline_version": "v1_baseline",
        "total_crops_processed": len(df_v1),
        "total_videos": len(tasks),
        "elapsed_seconds": round(elapsed, 2),
        "output_directory": to_relative_path(V1_OUTPUT_DIR),
        "manifest_path": to_relative_path(V1_MANIFEST_PATH)
    }
    with open(V1_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"v1 Baseline Batch Complete in {elapsed:.2f}s -> {V1_MANIFEST_PATH}")


if __name__ == "__main__":
    run_phase3_v1_batch()
