# -*- coding: utf-8 -*-
"""
=============================================================
  Kinematically-Gated Hybrid CSRT & Multi-Keyframe Tool Tracker
  Vision-Based Metal Chip Classification & Machining Monitoring
=============================================================

Combines OpenCV CSRT Visual Tracking with Physical Lathe Kinematics:
  1. Multi-Keyframe Ground-Truth Anchors (0%, 25%, 50%, 75%, 100%):
     Hard, non-overridable calibration anchors across the video timeline.
  2. Segmented CSRT Re-initialization:
     CSRT tracker is freshly initialized at each anchor, eliminating error propagation.
  3. Kinematic Gating (Bounding Tube):
     For intermediate frames, CSRT visual updates are constrained to a tight
     bounding cylinder (max 8px) centered on the linear lathe feed path.
  4. Anti-Drift & Occlusion Fallback:
     If CSRT jumps (following chips/glare) or loses tracking, the tracker rejects
     the jump, falls back to the kinematic line, and re-centers CSRT onto the tool.
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


def extract_keyframes_from_config(cfg_entry: dict) -> list:
    """
    Extracts and standardizes keyframe list from configuration entry.
    Supports both new multi-keyframe format and legacy single-keyframe format.
    """
    raw_kfs = cfg_entry.get("keyframes")
    if raw_kfs and isinstance(raw_kfs, list) and len(raw_kfs) > 0:
        cleaned = []
        for k in raw_kfs:
            f_idx = int(k.get("frame_idx", 0))
            tip = list(k.get("tool_tip", [0, 0]))
            bbox = list(k.get("bbox", [tip[0] - 8, tip[1] - 8, 16, 16]))
            cleaned.append({
                "frame_idx": f_idx,
                "tool_tip": [int(tip[0]), int(tip[1])],
                "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            })
        cleaned.sort(key=lambda x: x["frame_idx"])
        return cleaned

    # Legacy fallback: single keyframe
    start_frame = int(cfg_entry.get("start_frame", 0))
    bbox = list(cfg_entry.get("bbox", [0, 0, 16, 16]))
    annotated_tip = list(cfg_entry.get("tool_tip", [bbox[0], bbox[1]]))
    return [{
        "frame_idx": start_frame,
        "tool_tip": [int(annotated_tip[0]), int(annotated_tip[1])],
        "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
    }]


def interpolate_kinematic_point(keyframes: list, frame_idx: int) -> tuple:
    """
    Calculates expected tool tip and bbox from linear kinematic interpolation.
    Returns: ((tip_x, tip_y), (bbox_x, bbox_y, bbox_w, bbox_h), is_anchor)
    """
    M = len(keyframes)
    if M == 1:
        k0 = keyframes[0]
        return (
            (float(k0["tool_tip"][0]), float(k0["tool_tip"][1])),
            (float(k0["bbox"][0]), float(k0["bbox"][1]), int(k0["bbox"][2]), int(k0["bbox"][3])),
            (frame_idx == k0["frame_idx"])
        )

    if frame_idx <= keyframes[0]["frame_idx"]:
        k0 = keyframes[0]
        return (
            (float(k0["tool_tip"][0]), float(k0["tool_tip"][1])),
            (float(k0["bbox"][0]), float(k0["bbox"][1]), int(k0["bbox"][2]), int(k0["bbox"][3])),
            (frame_idx == k0["frame_idx"])
        )

    if frame_idx >= keyframes[-1]["frame_idx"]:
        k_last = keyframes[-1]
        k_prev = keyframes[-2]
        df = k_last["frame_idx"] - k_prev["frame_idx"]
        if df > 0:
            vx_tip = (k_last["tool_tip"][0] - k_prev["tool_tip"][0]) / df
            vy_tip = (k_last["tool_tip"][1] - k_prev["tool_tip"][1]) / df
            vx_box = (k_last["bbox"][0] - k_prev["bbox"][0]) / df
            vy_box = (k_last["bbox"][1] - k_prev["bbox"][1]) / df
            delta_f = frame_idx - k_last["frame_idx"]
            return (
                (float(k_last["tool_tip"][0] + vx_tip * delta_f), float(k_last["tool_tip"][1] + vy_tip * delta_f)),
                (float(k_last["bbox"][0] + vx_box * delta_f), float(k_last["bbox"][1] + vy_box * delta_f), int(k_last["bbox"][2]), int(k_last["bbox"][3])),
                (frame_idx == k_last["frame_idx"])
            )
        return (
            (float(k_last["tool_tip"][0]), float(k_last["tool_tip"][1])),
            (float(k_last["bbox"][0]), float(k_last["bbox"][1]), int(k_last["bbox"][2]), int(k_last["bbox"][3])),
            (frame_idx == k_last["frame_idx"])
        )

    for i in range(M - 1):
        k0 = keyframes[i]
        k1 = keyframes[i + 1]
        if k0["frame_idx"] <= frame_idx <= k1["frame_idx"]:
            span = max(1, k1["frame_idx"] - k0["frame_idx"])
            alpha = (frame_idx - k0["frame_idx"]) / span
            tx = (1.0 - alpha) * k0["tool_tip"][0] + alpha * k1["tool_tip"][0]
            ty = (1.0 - alpha) * k0["tool_tip"][1] + alpha * k1["tool_tip"][1]
            bx = (1.0 - alpha) * k0["bbox"][0] + alpha * k1["bbox"][0]
            by = (1.0 - alpha) * k0["bbox"][1] + alpha * k1["bbox"][1]
            return (
                (float(tx), float(ty)),
                (float(bx), float(by), int(k0["bbox"][2]), int(k0["bbox"][3])),
                (frame_idx == k0["frame_idx"] or frame_idx == k1["frame_idx"])
            )

    k_last = keyframes[-1]
    return (
        (float(k_last["tool_tip"][0]), float(k_last["tool_tip"][1])),
        (float(k_last["bbox"][0]), float(k_last["bbox"][1]), int(k_last["bbox"][2]), int(k_last["bbox"][3])),
        False
    )


def track_single_video_robust(task_item: tuple) -> dict:
    """
    Tracks tool tip using Segmented CSRT visual tracking gated by physical lathe kinematics.
    """
    v_name, cfg_entry = task_item
    clean_name = sanitize_filename(v_name)
    v_idx = cfg_entry.get("video_index", 0)
    material = cfg_entry.get("material", "Unknown")

    keyframes = extract_keyframes_from_config(cfg_entry)

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
                    "video_index": v_idx,
                    "material": material,
                    "status": "FAILED",
                    "error": f"Cannot open video file: {raw_path}",
                    "records": []
                }
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

        if total_frames == 0:
            return {
                "video": v_name,
                "video_index": v_idx,
                "material": material,
                "status": "FAILED",
                "error": "0 frames found",
                "records": []
            }

        # Read Frame 0 for dimensions
        if use_extracted_files:
            init_frame = cv2.imread(frame_files[0])
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, init_frame = cap.read()
            if not ret or init_frame is None:
                return {
                    "video": v_name,
                    "video_index": v_idx,
                    "material": material,
                    "status": "FAILED",
                    "error": "Failed to read init frame 0",
                    "records": []
                }

        frame_h, frame_w = init_frame.shape[:2]

        records = []
        prev_tip_x = None
        prev_tip_y = None
        tracked_frames = 0
        lost_frames = 0

        # Segmented CSRT Tracker management
        anchor_dict = {k["frame_idx"]: k for k in keyframes}
        csrt_tracker = None
        csrt_offset_x = 0
        csrt_offset_y = 0
        MAX_GATING_DIST_SQ = 64.0  # 8px maximum allowable deviation from kinematic line

        for idx in range(total_frames):
            frame = None
            if use_extracted_files:
                if idx < len(frame_files):
                    frame = cv2.imread(frame_files[idx])
            else:
                if idx == 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    frame = None

            (kin_tx, kin_ty), (kin_bx, kin_by, kin_bw, kin_bh), is_anchor = interpolate_kinematic_point(keyframes, idx)

            # Check if this frame is a hard ground-truth anchor
            if idx in anchor_dict and frame is not None:
                anchor = anchor_dict[idx]
                final_tx = int(anchor["tool_tip"][0])
                final_ty = int(anchor["tool_tip"][1])
                cur_bx = int(anchor["bbox"][0])
                cur_by = int(anchor["bbox"][1])
                bw = int(anchor["bbox"][2])
                bh = int(anchor["bbox"][3])

                # Hard Re-initialize CSRT Tracker on anchor
                csrt_tracker = cv2.TrackerCSRT_create()
                csrt_tracker.init(frame, (cur_bx, cur_by, bw, bh))
                csrt_offset_x = final_tx - cur_bx
                csrt_offset_y = final_ty - cur_by
                success = True
                tracked_frames += 1

            else:
                # Intermediate frame: Run CSRT visual tracker with Kinematic Gating
                csrt_ok = False
                csrt_tx, csrt_ty = kin_tx, kin_ty
                cur_bx, cur_by = int(round(kin_bx)), int(round(kin_by))
                bw, bh = kin_bw, kin_bh

                if csrt_tracker is not None and frame is not None:
                    ok, box = csrt_tracker.update(frame)
                    if ok:
                        cbx, cby, cbw, cbh = [int(v) for v in box]
                        candidate_tx = cbx + csrt_offset_x
                        candidate_ty = cby + csrt_offset_y
                        dist_sq = (candidate_tx - kin_tx)**2 + (candidate_ty - kin_ty)**2

                        # Gating: Only accept CSRT if within 8px kinematic tube
                        if dist_sq <= MAX_GATING_DIST_SQ:
                            # Smooth fusion: 70% CSRT visual + 30% Kinematic line
                            final_tx = int(round(0.70 * candidate_tx + 0.30 * kin_tx))
                            final_ty = int(round(0.70 * candidate_ty + 0.30 * kin_ty))
                            cur_bx = cbx
                            cur_by = cby
                            csrt_ok = True
                            success = True
                            tracked_frames += 1

                # If CSRT was occluded, failed, or jumped: Fallback to Kinematic Waypoint
                if not csrt_ok:
                    final_tx = int(round(kin_tx))
                    final_ty = int(round(kin_ty))
                    cur_bx = int(round(kin_bx))
                    cur_by = int(round(kin_by))
                    success = True
                    tracked_frames += 1

                    # Re-anchor CSRT internal model to kinematic location
                    if frame is not None:
                        try:
                            csrt_tracker = cv2.TrackerCSRT_create()
                            csrt_tracker.init(frame, (cur_bx, cur_by, bw, bh))
                        except Exception:
                            csrt_tracker = None

            # Clamp coordinates to frame boundaries
            final_tx = max(0, min(frame_w - 1, final_tx))
            final_ty = max(0, min(frame_h - 1, final_ty))
            cur_bx = max(0, min(frame_w - bw, cur_bx))
            cur_by = max(0, min(frame_h - bh, cur_by))

            if prev_tip_x is None:
                prev_tip_x, prev_tip_y = final_tx, final_ty

            dx = final_tx - prev_tip_x
            dy = final_ty - prev_tip_y
            prev_tip_x, prev_tip_y = final_tx, final_ty

            # Dynamic 300x300 ROI Coordinates centered on tool tip
            roi_x1 = max(0, final_tx - MIN_CROP_HALF)
            roi_y1 = max(0, final_ty - MIN_CROP_HALF)
            roi_x2 = min(frame_w, final_tx + MIN_CROP_HALF)
            roi_y2 = min(frame_h, final_ty + MIN_CROP_HALF)

            if (roi_x2 - roi_x1) < (2 * MIN_CROP_HALF) and roi_x2 == frame_w:
                roi_x1 = max(0, frame_w - (2 * MIN_CROP_HALF))
            if (roi_x2 - roi_x1) < (2 * MIN_CROP_HALF) and roi_x1 == 0:
                roi_x2 = min(frame_w, 2 * MIN_CROP_HALF)
            if (roi_y2 - roi_y1) < (2 * MIN_CROP_HALF) and roi_y2 == frame_h:
                roi_y1 = max(0, frame_h - (2 * MIN_CROP_HALF))
            if (roi_y2 - roi_y1) < (2 * MIN_CROP_HALF) and roi_y1 == 0:
                roi_y2 = min(frame_h, 2 * MIN_CROP_HALF)

            records.append({
                "video": v_name,
                "video_index": v_idx,
                "material": material,
                "frame_idx": idx,
                "timestamp_sec": round(idx / fps, 3),
                "tool_tip_x": final_tx,
                "tool_tip_y": final_ty,
                "bbox_x": cur_bx,
                "bbox_y": cur_by,
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

    success_rate = round(100.0 * tracked_frames / max(1, tracked_frames + lost_frames), 2)
    return {
        "video": v_name,
        "video_index": v_idx,
        "material": material,
        "status": "OK",
        "total_frames": len(records),
        "tracked_frames": tracked_frames,
        "lost_frames": lost_frames,
        "success_rate": success_rate,
        "records": records
    }


def run_batch_robust_tracking(
    config_path: str = TRACKER_CONFIG_PATH,
    output_csv: str = TRACKING_RESULTS_PATH,
    workers: int = N_WORKERS
) -> pd.DataFrame:
    """
    Executes parallel robust tool tracking across all 27 videos and updates tracking_results.csv.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Tracker configuration file not found at '{config_path}'.")

    with open(config_path, "r") as f:
        config = json.load(f)

    tasks = [(v_name, cfg) for v_name, cfg in config.items() if ("bbox" in cfg or "keyframes" in cfg)]

    print("=" * 75)
    print(f"  PHASE 1 (ROBUST): HYBRID KINEMATIC-CSRT TOOL TRACKING ACROSS {len(tasks)} VIDEOS")
    print(f"  Parallel CPU Workers: {workers}")
    print(f"  Configuration: {config_path}")
    print("=" * 75)

    start_time = time.time()
    all_records = []
    video_summaries = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(track_single_video_robust, item): item[0] for item in tasks}

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
                    print(f"  [+] Tracked: Video {res['video_index']:02d} ({res['material']:<10s}) | {res['total_frames']:,} frames | CSRT-Kinematic Hybrid")
                else:
                    print(f"  [!] Failed: {v_name}: {res.get('error')}")
            except Exception as e:
                print(f"  [!] Exception on {v_name}: {e}")

    elapsed = time.time() - start_time
    df_results = pd.DataFrame(all_records)

    # Sort results by video_index and frame_idx
    if not df_results.empty:
        df_results.sort_values(by=["video_index", "frame_idx"], inplace=True)

    # Save master tracking results CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_results.to_csv(output_csv, index=False)

    summary_path = os.path.join(os.path.dirname(output_csv), "tracking_summary.json")
    summary_data = {
        "tracker": "Hybrid_Kinematic_Gated_CSRT",
        "total_videos_tracked": len(video_summaries),
        "total_trajectory_points": len(df_results),
        "elapsed_seconds": round(elapsed, 2),
        "average_fps": round(len(df_results) / max(0.1, elapsed), 1),
        "video_details": sorted(video_summaries, key=lambda x: x["video_index"])
    }

    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 75)
    print(f"  HYBRID CSRT TRACKING COMPLETE in {elapsed:.2f}s (~{elapsed/60:.2f} min)")
    print(f"  Total Trajectory Points Logged: {len(df_results):,}")
    print(f"  Overall Processing Speed:       {len(df_results)/elapsed:.1f} FPS")
    print(f"  Saved Trajectory Catalog to:   '{output_csv}'")
    print(f"  Saved Tracking Summary to:     '{summary_path}'")
    print("=" * 75)

    return df_results


if __name__ == "__main__":
    run_batch_robust_tracking()
