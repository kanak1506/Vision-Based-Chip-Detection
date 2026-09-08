# -*- coding: utf-8 -*-
"""
=============================================================
  Phase 0: Video Cataloging & Metadata Extraction Module
=============================================================
"""

import os
import cv2
import pandas as pd
from src.config import RAW_VIDEOS_DIR, VIDEO_METADATA_PATH


def extract_video_metadata(video_dir=RAW_VIDEOS_DIR, output_csv=VIDEO_METADATA_PATH):
    """
    Scans the raw video directory, audits frame counts, resolution, FPS,
    durations, and codec status, and saves the metadata catalog to CSV.
    """
    video_files = [
        f for f in os.listdir(video_dir)
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]
    video_files.sort()

    records = []
    print(f"Auditing {len(video_files)} video files in '{video_dir}'...\n")

    for idx, fname in enumerate(video_files, start=1):
        fpath = os.path.join(video_dir, fname)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)

        cap = cv2.VideoCapture(fpath)
        if not cap.isOpened():
            print(f"[{idx}] ERROR opening: {fname}")
            records.append({
                "Index": idx,
                "Filename": fname,
                "Size_MB": round(size_mb, 2),
                "Status": "Corrupted / Cannot Open",
                "Width": None,
                "Height": None,
                "FPS": None,
                "Total_Frames": None,
                "Duration_Sec": None,
                "Codec": None
            })
            continue

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = (total_frames / fps) if (fps and fps > 0) else 0
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

            ret, _ = cap.read()
            status = "OK" if ret else "Read Error on Frame 0"
        finally:
            cap.release()

        records.append({
            "Index": idx,
            "Filename": fname,
            "Size_MB": round(size_mb, 2),
            "Status": status,
            "Width": width,
            "Height": height,
            "FPS": round(fps, 2),
            "Total_Frames": total_frames,
            "Duration_Sec": round(duration_sec, 2),
            "Codec": fourcc.strip()
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Successfully saved video metadata catalog to '{output_csv}'.")

    print("\n=== VIDEO METADATA SUMMARY ===")
    print(f"Total Videos: {len(df)}")
    print(f"All Videos Readable: {(df['Status'] == 'OK').all()}")
    print(f"Total Duration: {round(df['Duration_Sec'].sum(), 2)}s (~{round(df['Duration_Sec'].sum() / 60, 2)} min)")
    print(f"Total Frames: {df['Total_Frames'].sum():,}")
    print(f"Total Disk Size: {round(df['Size_MB'].sum(), 2)} MB")

    return df


if __name__ == "__main__":
    extract_video_metadata()
