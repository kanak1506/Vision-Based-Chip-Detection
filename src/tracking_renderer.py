# -*- coding: utf-8 -*-
"""
=============================================================
  Phase 1: Tool Tip Tracking Overlay Frame Renderer Module
  Vision-Based Metal Chip Classification & Machining Monitoring
=============================================================
"""

import os
import glob
import time
import json
import cv2
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.config import (
    RAW_VIDEOS_DIR,
    EXTRACTED_FRAMES_DIR,
    PHASE1_TRACKED_DIR,
    TRACKING_RESULTS_PATH,
    MIN_CROP_HALF,
    sanitize_filename,
    draw_tool_tip_blue_dot,
    to_relative_path,
    N_WORKERS
)


def render_video_tracked_frames(args: tuple) -> dict:
    """
    Renders tracking overlays (Blue Dot, CSRT BBox, 300x300 ROI) for a single video.
    """
    v_name, v_records, output_base_dir, jpeg_quality = args
    clean_name = sanitize_filename(v_name)
    target_dir = os.path.join(output_base_dir, clean_name)
    os.makedirs(target_dir, exist_ok=True)

    extracted_dir = os.path.join(EXTRACTED_FRAMES_DIR, clean_name)
    frame_files = sorted(glob.glob(os.path.join(extracted_dir, "frame_*.jpg")))
    use_extracted_files = len(frame_files) > 0

    cap = None
    try:
        if not use_extracted_files:
            raw_path = os.path.join(RAW_VIDEOS_DIR, v_name)
            cap = cv2.VideoCapture(raw_path)
            if not cap.isOpened():
                return {"video": v_name, "status": "ERROR", "frames_rendered": 0, "error": "Cannot open video"}

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        rendered_count = 0
        total_frames = len(v_records)

        v_idx = v_records[0].get("video_index", 0)
        material = v_records[0].get("material", "Unknown")

        for row in v_records:
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
            display = frame.copy()

            tx = int(row["tool_tip_x"])
            ty = int(row["tool_tip_y"])
            bx = int(row.get("bbox_x", tx - 10))
            by = int(row.get("bbox_y", ty - 10))
            bw = int(row.get("bbox_w", 20))
            bh = int(row.get("bbox_h", 20))
            rx1 = int(row.get("roi_x1", max(0, tx - MIN_CROP_HALF)))
            ry1 = int(row.get("roi_y1", max(0, ty - MIN_CROP_HALF)))
            rx2 = int(row.get("roi_x2", min(W, tx + MIN_CROP_HALF)))
            ry2 = int(row.get("roi_y2", min(H, ty + MIN_CROP_HALF)))

            # 1. Dynamic 300x300 ROI Box (Yellow / Gold)
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (0, 220, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(display, "300x300 ROI", (rx1 + 6, ry1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

            # 2. CSRT Bounding Box (Cyan)
            cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (255, 255, 0), 1, lineType=cv2.LINE_AA)

            # 3. Tool Tip Blue Dot removed per user preference (coordinates preserved in metadata)

            # 4. Top Info Header
            cv2.rectangle(display, (0, 0), (W, 36), (15, 15, 15), -1)
            header = f"Video {v_idx:02d} ({material}) | Frame: {f_idx + 1:06d}/{total_frames:06d} | Tool Tip: ({tx}, {ty})"
            cv2.putText(display, header, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            # Save annotated frame
            out_fpath = os.path.join(target_dir, f"frame_{f_idx:06d}.jpg")
            cv2.imwrite(out_fpath, display, encode_param)
            rendered_count += 1
    finally:
        if cap is not None:
            cap.release()

    return {
        "video": v_name,
        "clean_name": clean_name,
        "status": "OK",
        "frames_rendered": rendered_count,
        "folder": to_relative_path(target_dir)
    }


def run_phase1_frame_rendering(
    tracking_csv: str = TRACKING_RESULTS_PATH,
    output_dir: str = PHASE1_TRACKED_DIR,
    workers: int = N_WORKERS,
    jpeg_quality: int = 95
) -> dict:
    """
    Renders all 62,301 frames with tracking overlays across all 27 videos in parallel.
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    df_tracking = pd.read_csv(tracking_csv)
    video_groups = list(df_tracking.groupby("video"))

    print("=" * 75)
    print(f"  PHASE 1: RENDERING TRACKED OVERLAY FRAMES ({len(df_tracking):,} TOTAL FRAMES)")
    print(f"  Total Videos:     {len(video_groups)}")
    print(f"  Output Directory: {output_dir}")
    print(f"  JPEG Quality:     {jpeg_quality}")
    print(f"  Workers:          {workers}")
    print("=" * 75)

    tasks = [
        (v_name, group.to_dict("records"), output_dir, jpeg_quality)
        for v_name, group in video_groups
    ]

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(render_video_tracked_frames, t): t[0] for t in tasks}

        for future in as_completed(futures):
            v_name = futures[future]
            try:
                res = future.result()
                results.append(res)
                print(f"  [+] Rendered: {res['clean_name']} -> {res['frames_rendered']:,} frames")
            except Exception as e:
                print(f"  [!] Failed: {v_name}: {e}")
                results.append({"video": v_name, "status": "FAILED", "error": str(e)})

    elapsed = time.time() - start_time
    total_rendered = sum(r.get("frames_rendered", 0) for r in results)

    # Save summary
    summary_path = os.path.join(os.path.dirname(output_dir), "metadata", "phase1_rendering_summary.json")
    summary = {
        "total_videos": len(video_groups),
        "total_frames_rendered": total_rendered,
        "elapsed_seconds": round(elapsed, 2),
        "output_directory": to_relative_path(output_dir),
        "details": results
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 75)
    print(f"  RENDERING COMPLETE in {elapsed:.2f}s (~{elapsed/60:.2f} min)")
    print(f"  Total Tracked Frames Saved: {total_rendered:,}")
    print(f"  Saved Summary to:           '{summary_path}'")
    print("=" * 75)

    return summary


if __name__ == "__main__":
    run_phase1_frame_rendering()
