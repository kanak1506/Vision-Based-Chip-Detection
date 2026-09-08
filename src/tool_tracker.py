# -*- coding: utf-8 -*-
"""
=============================================================
  Phase 1: CSRT Tool Tip Tracking & Dynamic ROI Logging
  Vision-Based Metal Chip Classification & Machining Monitoring
=============================================================
"""

import os
import glob
import json
import time
import pandas as pd
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.config import (
    RAW_VIDEOS_DIR,
    EXTRACTED_FRAMES_DIR,
    TRACKER_CONFIG_PATH,
    TRACKING_RESULTS_PATH,
    MIN_CROP_HALF,
    sanitize_filename,
    N_WORKERS
)


def track_single_video(task_item: tuple) -> dict:
    """
    Worker function to track tool tip on a single video using OpenCV CSRT.
    """
    v_name, cfg_entry = task_item
    clean_name = sanitize_filename(v_name)
    
    start_frame = cfg_entry.get("start_frame", 0)
    bbox = tuple(cfg_entry.get("bbox", (0, 0, 0, 0)))
    offset = tuple(cfg_entry.get("offset_from_bbox", (0, 0)))
    v_idx = cfg_entry.get("video_index", 0)
    material = cfg_entry.get("material", "Unknown")

    # Discover frame source
    extracted_dir = os.path.join(EXTRACTED_FRAMES_DIR, clean_name)
    frame_files = sorted(glob.glob(os.path.join(extracted_dir, "frame_*.jpg")))
    use_extracted_files = len(frame_files) > 0

    cap = None
    try:
        total_frames = 0
        fps = 60.0

        if use_extracted_files:
            total_frames = len(frame_files)
        else:
            raw_path = os.path.join(RAW_VIDEOS_DIR, v_name)
            cap = cv2.VideoCapture(raw_path)
            if not cap.isOpened():
                return {
                    "video": v_name,
                    "status": "FAILED",
                    "error": f"Cannot open video file: {raw_path}",
                    "records": []
                }
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

        # 1. Read Start Frame for Tracker Initialization
        if use_extracted_files:
            if start_frame >= len(frame_files):
                start_frame = 0
            init_frame = cv2.imread(frame_files[start_frame])
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, init_frame = cap.read()
            if not ret:
                return {
                    "video": v_name,
                    "status": "FAILED",
                    "error": f"Cannot read start frame {start_frame}",
                    "records": []
                }

        if init_frame is None:
            return {
                "video": v_name,
                "status": "FAILED",
                "error": "Init frame is None",
                "records": []
            }

        frame_h, frame_w = init_frame.shape[:2]

        # 2. Initialize CSRT Tracker
        tracker = cv2.TrackerCSRT_create()
        tracker.init(init_frame, bbox)

        records = []
        prev_tip_x = int(bbox[0] + offset[0])
        prev_tip_y = int(bbox[1] + offset[1])
        lost_frames = 0
        tracked_frames = 0

        # 3. Track Forward Through All Frames
        for idx in range(total_frames):
            if use_extracted_files:
                frame = cv2.imread(frame_files[idx])
                if frame is None:
                    continue
            else:
                if idx == 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break

            # If before start_frame, use static initial position
            if idx < start_frame:
                tip_x, tip_y = prev_tip_x, prev_tip_y
                success = True
                bx, by, bw, bh = bbox
            else:
                success, new_bbox = tracker.update(frame)
                if success:
                    bx, by, bw, bh = [int(v) for v in new_bbox]
                    tip_x = int(bx + offset[0])
                    tip_y = int(by + offset[1])
                    tracked_frames += 1
                else:
                    lost_frames += 1
                    tip_x, tip_y = prev_tip_x, prev_tip_y
                    bx, by, bw, bh = tip_x - offset[0], tip_y - offset[1], bbox[2], bbox[3]

            # Velocity deltas
            dx = tip_x - prev_tip_x
            dy = tip_y - prev_tip_y
            prev_tip_x, prev_tip_y = tip_x, tip_y

            # Dynamic 300x300 ROI Coordinates
            roi_x1 = max(0, tip_x - MIN_CROP_HALF)
            roi_y1 = max(0, tip_y - MIN_CROP_HALF)
            roi_x2 = min(frame_w, tip_x + MIN_CROP_HALF)
            roi_y2 = min(frame_h, tip_y + MIN_CROP_HALF)

            records.append({
                "video": v_name,
                "video_index": v_idx,
                "material": material,
                "frame_idx": idx,
                "timestamp_sec": round(idx / fps, 3),
                "tool_tip_x": tip_x,
                "tool_tip_y": tip_y,
                "bbox_x": bx,
                "bbox_y": by,
                "bbox_w": bw,
                "bbox_h": bh,
                "roi_x1": roi_x1,
                "roi_y1": roi_y1,
                "roi_x2": roi_x2,
                "roi_y2": roi_y2,
                "delta_x": dx,
                "delta_y": dy,
                "tracking_status": "TRACKED" if success else "LOST"
            })
    finally:
        if cap is not None:
            cap.release()

    return {
        "video": v_name,
        "video_index": v_idx,
        "material": material,
        "status": "OK",
        "total_frames": len(records),
        "tracked_frames": tracked_frames,
        "lost_frames": lost_frames,
        "success_rate": round(100.0 * tracked_frames / max(1, tracked_frames + lost_frames), 2),
        "records": records
    }


def run_batch_tracking(
    config_path: str = TRACKER_CONFIG_PATH,
    output_csv: str = TRACKING_RESULTS_PATH,
    workers: int = N_WORKERS
) -> pd.DataFrame:
    """
    Executes parallel CSRT tracking across all configured videos and saves master trajectory dataset.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Tracker configuration file not found at '{config_path}'. Please run calibrate_tool_tip.py first.")

    with open(config_path, "r") as f:
        config = json.load(f)

    tasks = [(v_name, cfg) for v_name, cfg in config.items() if "bbox" in cfg]

    print("=" * 75)
    print(f"  PHASE 1: BATCH CSRT TOOL TRACKING ACROSS {len(tasks)} VIDEOS")
    print(f"  Parallel CPU Workers: {workers}")
    print(f"  Configuration: {config_path}")
    print("=" * 75)

    start_time = time.time()
    all_records = []
    video_summaries = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(track_single_video, item): item[0] for item in tasks}

        for future in as_completed(futures):
            v_name = futures[future]
            try:
                res = future.result()
                if res["status"] == "OK":
                    all_records.extend(res["records"])
                    video_summaries.append({
                        "video": res["video"],
                        "video_index": res["video_index"],
                        "material": res["material"],
                        "total_frames": res["total_frames"],
                        "tracked_frames": res["tracked_frames"],
                        "lost_frames": res["lost_frames"],
                        "success_rate": res["success_rate"]
                    })
                    print(f"  [+] Tracked: Video {res['video_index']:02d} ({res['material']:<10s}) | {res['total_frames']:,} frames | Success: {res['success_rate']}%")
                else:
                    print(f"  [!] Failed: {v_name}: {res.get('error')}")
            except Exception as e:
                print(f"  [!] Exception on {v_name}: {e}")

    elapsed = time.time() - start_time
    df_results = pd.DataFrame(all_records)

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_results.to_csv(output_csv, index=False)

    # Save tracking summary JSON
    summary_path = os.path.join(os.path.dirname(output_csv), "tracking_summary.json")
    summary_data = {
        "total_videos_tracked": len(video_summaries),
        "total_trajectory_points": len(df_results),
        "elapsed_seconds": round(elapsed, 2),
        "average_fps": round(len(df_results) / max(0.1, elapsed), 1),
        "video_details": sorted(video_summaries, key=lambda x: x["video_index"])
    }

    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 75)
    print(f"  TRACKING PIPELINE COMPLETE in {elapsed:.2f}s (~{elapsed/60:.2f} min)")
    print(f"  Total Trajectory Points Logged: {len(df_results):,}")
    print(f"  Overall Processing Speed:       {len(df_results)/elapsed:.1f} FPS")
    print(f"  Saved Trajectory Catalog to:   '{output_csv}'")
    print(f"  Saved Tracking Summary to:     '{summary_path}'")
    print("=" * 75)

    return df_results


if __name__ == "__main__":
    run_batch_tracking()
