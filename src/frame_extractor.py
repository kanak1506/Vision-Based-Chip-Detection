# -*- coding: utf-8 -*-
"""
=============================================================
  Phase 0: Multi-threaded Video Frame Extraction Module
  Vision-Based Metal Chip Classification & Machining Monitoring
=============================================================
"""

import os
import cv2
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.config import (
    RAW_VIDEOS_DIR,
    EXTRACTED_FRAMES_DIR,
    sanitize_filename,
    to_relative_path,
    N_WORKERS
)


def extract_frames_from_video(video_path: str, output_dir: str, jpeg_quality: int = 95) -> dict:
    """
    Extracts all frames from a single video into a dedicated subfolder.
    """
    v_name = os.path.basename(video_path)
    clean_name = sanitize_filename(v_name)
    target_folder = os.path.join(output_dir, clean_name)
    os.makedirs(target_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "video": v_name,
            "status": "ERROR",
            "frames_extracted": 0,
            "folder": target_folder,
            "error": "Could not open video capture."
        }

    try:
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_idx = 0
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            out_path = os.path.join(target_folder, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(out_path, frame, encode_param)
            frame_idx += 1
    finally:
        cap.release()

    return {
        "video": v_name,
        "clean_name": clean_name,
        "status": "OK",
        "frames_extracted": frame_idx,
        "expected_frames": total_video_frames,
        "fps": round(fps, 2),
        "resolution": f"{width}x{height}",
        "folder": to_relative_path(target_folder)
    }


def run_frame_extraction(
    raw_videos_dir: str = RAW_VIDEOS_DIR,
    output_dir: str = EXTRACTED_FRAMES_DIR,
    workers: int = N_WORKERS,
    jpeg_quality: int = 95
) -> dict:
    """
    Extracts all frames from all videos in parallel.
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    video_files = [
        os.path.join(raw_videos_dir, f)
        for f in sorted(os.listdir(raw_videos_dir))
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]

    print("=" * 70)
    print(f"  PHASE 0: EXTRACTING FRAMES FROM {len(video_files)} VIDEOS")
    print(f"  Output Directory: {output_dir}")
    print(f"  Workers: {workers} | JPEG Quality: {jpeg_quality}")
    print("=" * 70)

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(extract_frames_from_video, v_path, output_dir, jpeg_quality): v_path
            for v_path in video_files
        }

        for future in as_completed(futures):
            v_path = futures[future]
            try:
                res = future.result()
                results.append(res)
                print(f"  [+] Extracted: {res['clean_name']} -> {res['frames_extracted']} frames")
            except Exception as e:
                print(f"  [!] Failed: {os.path.basename(v_path)}: {e}")
                results.append({
                    "video": os.path.basename(v_path),
                    "status": "FAILED",
                    "error": str(e)
                })

    elapsed = time.time() - start_time
    total_frames = sum(r.get("frames_extracted", 0) for r in results)

    # Save summary
    summary_path = os.path.join(os.path.dirname(output_dir), "metadata", "frame_extraction_summary.json")
    summary = {
        "total_videos": len(video_files),
        "total_frames_extracted": total_frames,
        "elapsed_seconds": round(elapsed, 2),
        "output_directory": to_relative_path(output_dir),
        "details": results
    }

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  EXTRACTION COMPLETE in {elapsed:.2f}s (~{elapsed/60:.2f} min)")
    print(f"  Total Frames Extracted: {total_frames:,}")
    print(f"  Saved Summary to: {summary_path}")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    run_frame_extraction()
