# -*- coding: utf-8 -*-
"""
=============================================================
  Phase 2: Sub-sampling (~4 FPS), Quality Tagging & ROI Extraction
  Vision-Based Metal Chip Classification & Machining Monitoring
=============================================================
"""

import os
import glob
import time
import json
import shutil
import cv2
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.config import (
    RAW_VIDEOS_DIR,
    EXTRACTED_FRAMES_DIR,
    PHASE2_CROPS_DIR,
    TRACKING_RESULTS_PATH,
    SPLITS_VIDEO_WISE_PATH,
    PHASE2_MANIFEST_PATH,
    PHASE2_SUMMARY_PATH,
    SAMPLE_EVERY_N,
    MIN_CROP_HALF,
    sanitize_filename,
    N_WORKERS
)


def compute_laplacian_var(image_bgr: np.ndarray) -> float:
    """Computes sharpness score using variance of the Laplacian."""
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_motion_score(curr_bgr: np.ndarray, prev_bgr: np.ndarray) -> float:
    """Computes frame-to-frame dynamic motion score from pixel differences."""
    if curr_bgr is None or prev_bgr is None or curr_bgr.shape != prev_bgr.shape:
        return 0.0
    gray_curr = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_curr, gray_prev)
    return float(np.mean(diff))


def process_video_subsampling(args: tuple) -> dict:
    """
    Subsamples a single video at ~4 FPS, tags sharpness & motion, and extracts 300x300 ROI crops.
    """
    v_name, v_df_records, split_name, output_base_dir, sample_rate = args
    clean_name = sanitize_filename(v_name)

    # Auto-cleanup previous run output folders for this video
    retained_dir = os.path.join(output_base_dir, "retained", clean_name)
    negative_dir = os.path.join(output_base_dir, "negative_no_tool", clean_name)
    if os.path.exists(retained_dir):
        shutil.rmtree(retained_dir, ignore_errors=True)
    if os.path.exists(negative_dir):
        shutil.rmtree(negative_dir, ignore_errors=True)

    extracted_dir = os.path.join(EXTRACTED_FRAMES_DIR, clean_name)
    frame_files = sorted(glob.glob(os.path.join(extracted_dir, "frame_*.jpg")))
    use_extracted_files = len(frame_files) > 0

    cap = None
    try:
        if not use_extracted_files:
            raw_path = os.path.join(RAW_VIDEOS_DIR, v_name)
            cap = cv2.VideoCapture(raw_path)

        v_idx = v_df_records[0].get("video_index", 0)
        material = v_df_records[0].get("material", "Unknown")

        # Sample every Nth frame
        sampled_records = v_df_records[::sample_rate]

        crop_metadata_list = []
        prev_crop = None

        for row in sampled_records:
            f_idx = int(row["frame_idx"])

            if use_extracted_files:
                if f_idx < len(frame_files):
                    frame = cv2.imread(frame_files[f_idx])
                else:
                    frame = None
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                frame = frame if ret else None

            if frame is None:
                continue

            H, W = frame.shape[:2]
            tx = int(row["tool_tip_x"])
            ty = int(row["tool_tip_y"])
            rx1 = int(row.get("roi_x1", max(0, tx - MIN_CROP_HALF)))
            ry1 = int(row.get("roi_y1", max(0, ty - MIN_CROP_HALF)))
            rx2 = int(row.get("roi_x2", min(W, tx + MIN_CROP_HALF)))
            ry2 = int(row.get("roi_y2", min(H, ty + MIN_CROP_HALF)))
            status = str(row.get("tracking_status", "TRACKED"))

            # Extract 300x300 Crop
            crop = frame[ry1:ry2, rx1:rx2].copy()
            if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            # Compute Quality Metrics
            lap_var = compute_laplacian_var(crop)
            motion = compute_motion_score(crop, prev_crop) if prev_crop is not None else 0.0
            prev_crop = crop.copy()

            # Calculate Tool Tip in Crop Coordinates (relative to top-left of crop)
            crop_tip_x = int(tx - rx1)
            crop_tip_y = int(ty - ry1)

            # Classification category (RETAINED vs NO_TOOL_BACKGROUND)
            if status == "TRACKED" and crop.shape[0] >= 150 and crop.shape[1] >= 150:
                category = "RETAINED"
                target_folder = os.path.join(output_base_dir, "retained", clean_name)
                rel_folder = os.path.join("data", "phase2_roi_crops", "retained", clean_name)
            else:
                category = "NO_TOOL_BACKGROUND"
                target_folder = os.path.join(output_base_dir, "negative_no_tool", clean_name)
                rel_folder = os.path.join("data", "phase2_roi_crops", "negative_no_tool", clean_name)

            os.makedirs(target_folder, exist_ok=True)
            crop_filename = f"crop_{f_idx:06d}.jpg"
            crop_save_path = os.path.join(target_folder, crop_filename)
            rel_crop_path = os.path.join(rel_folder, crop_filename)
            cv2.imwrite(crop_save_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            crop_metadata_list.append({
                "video": v_name,
                "video_index": v_idx,
                "material": material,
                "split": split_name,
                "frame_idx": f_idx,
                "timestamp_sec": float(row.get("timestamp_sec", 0.0)),
                "tool_tip_x": tx,
                "tool_tip_y": ty,
                "crop_tip_x": crop_tip_x,
                "crop_tip_y": crop_tip_y,
                "roi_x1": rx1,
                "roi_y1": ry1,
                "roi_x2": rx2,
                "roi_y2": ry2,
                "crop_w": crop.shape[1],
                "crop_h": crop.shape[0],
                "laplacian_var": round(lap_var, 2),
                "motion_score": round(motion, 2),
                "tracking_status": status,
                "category": category,
                "crop_path": rel_crop_path.replace("\\", "/")
            })
    finally:
        if cap is not None:
            cap.release()

    return {
        "video": v_name,
        "clean_name": clean_name,
        "material": material,
        "split": split_name,
        "status": "OK",
        "total_sampled_crops": len(crop_metadata_list),
        "retained_valid_crops": sum(1 for c in crop_metadata_list if c["category"] == "RETAINED"),
        "no_tool_background_crops": sum(1 for c in crop_metadata_list if c["category"] == "NO_TOOL_BACKGROUND"),
        "crop_metadata": crop_metadata_list
    }


def run_subsampling_pipeline(
    tracking_csv: str = TRACKING_RESULTS_PATH,
    splits_csv: str = SPLITS_VIDEO_WISE_PATH,
    output_crops_dir: str = PHASE2_CROPS_DIR,
    output_manifest_csv: str = PHASE2_MANIFEST_PATH,
    output_summary_json: str = PHASE2_SUMMARY_PATH,
    sample_every_n: int = SAMPLE_EVERY_N,
    workers: int = N_WORKERS
) -> pd.DataFrame:
    """
    Executes parallel Phase 2 frame subsampling (~4 FPS), quality tagging, and ROI crop extraction.
    """
    if not os.path.exists(tracking_csv):
        raise FileNotFoundError(f"Tracking CSV not found at '{tracking_csv}'. Run Phase 1 tracking first.")

    start_time = time.time()
    os.makedirs(output_crops_dir, exist_ok=True)

    df_tracking = pd.read_csv(tracking_csv)

    # Load video-wise split mapping
    split_map = {}
    if os.path.exists(splits_csv):
        df_splits = pd.read_csv(splits_csv)
        split_map = dict(zip(df_splits["Filename"], df_splits["Split"]))

    video_tasks = []
    for v_name, group in df_tracking.groupby("video"):
        records = group.sort_values("frame_idx").to_dict("records")
        v_split = split_map.get(v_name, "train")
        video_tasks.append((v_name, records, v_split, output_crops_dir, sample_every_n))

    print("=" * 75)
    print(f"  PHASE 2: FRAME SUBSAMPLING (~4 FPS) & ROI CROP EXTRACTION")
    print(f"  Sampling Rate: Every {sample_every_n}th frame (~4 FPS)")
    print(f"  Total Videos:  {len(video_tasks)}")
    print(f"  Parallel CPU Workers: {workers}")
    print("=" * 75)

    all_crop_records = []
    video_summaries = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_video_subsampling, task): task[0] for task in video_tasks}

        for future in as_completed(futures):
            v_name = futures[future]
            try:
                res = future.result()
                if res["status"] == "OK":
                    all_crop_records.extend(res["crop_metadata"])
                    video_summaries.append({
                        "video": res["video"],
                        "clean_name": res["clean_name"],
                        "material": res["material"],
                        "split": res["split"],
                        "total_sampled_crops": res["total_sampled_crops"],
                        "retained_valid_crops": res["retained_valid_crops"],
                        "no_tool_background_crops": res["no_tool_background_crops"]
                    })
                    print(f"  [+] Extracted: {res['clean_name']} | {res['retained_valid_crops']} valid ROI crops ({res['split'].upper()})")
                else:
                    print(f"  [!] Failed: {v_name}")
            except Exception as e:
                print(f"  [!] Error on {v_name}: {e}")

    # Clean up empty subdirectories and orphans
    for root_cat in ["retained", "negative_no_tool"]:
        cat_dir = os.path.join(output_crops_dir, root_cat)
        if os.path.exists(cat_dir):
            for sub in os.listdir(cat_dir):
                sub_path = os.path.join(cat_dir, sub)
                if os.path.isdir(sub_path):
                    # If folder is empty, delete it
                    if not os.listdir(sub_path):
                        shutil.rmtree(sub_path, ignore_errors=True)

    elapsed = time.time() - start_time
    df_manifest = pd.DataFrame(all_crop_records)

    # Save Master Manifest CSV
    os.makedirs(os.path.dirname(output_manifest_csv), exist_ok=True)
    df_manifest.to_csv(output_manifest_csv, index=False)

    # Compile Summary JSON
    summary_data = {
        "dataset_statistics": {
            "total_raw_frames": len(df_tracking),
            "sample_rate": sample_every_n,
            "total_sampled_crops": len(df_manifest),
            "retained_valid_crops": int((df_manifest["category"] == "RETAINED").sum()),
            "no_tool_background_crops": int((df_manifest["category"] == "NO_TOOL_BACKGROUND").sum()),
            "elapsed_seconds": round(elapsed, 2)
        },
        "material_distribution": df_manifest.groupby(["material", "category"]).size().unstack(fill_value=0).to_dict(),
        "split_distribution": df_manifest[df_manifest["category"] == "RETAINED"].groupby(["material", "split"]).size().unstack(fill_value=0).to_dict(),
        "video_summaries": sorted(video_summaries, key=lambda x: x["video"])
    }

    with open(output_summary_json, "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 75)
    print(f"  SUBSAMPLING COMPLETE in {elapsed:.2f}s (~{elapsed/60:.2f} min)")
    print(f"  Total Curated Crops Extracted: {len(df_manifest):,}")
    print(f"    - Valid Machining Crops:     {(df_manifest['category'] == 'RETAINED').sum():,}")
    print(f"    - No-Tool Background Crops:  {(df_manifest['category'] == 'NO_TOOL_BACKGROUND').sum():,}")
    print(f"  Saved Manifest to:             '{output_manifest_csv}'")
    print(f"  Saved Summary to:              '{output_summary_json}'")
    print("=" * 75)

    return df_manifest


if __name__ == "__main__":
    run_subsampling_pipeline()
