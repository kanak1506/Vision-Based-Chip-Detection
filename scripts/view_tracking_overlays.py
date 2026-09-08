# -*- coding: utf-8 -*-
"""
=============================================================
  Tool Tip Tracking Overlay & Visual Verification Player
  Vision-Based Metal Chip Classification & Machining Monitoring
=============================================================

Renders the CSRT tracked Tool Tip as a bright BLUE DOT on every frame
with the dynamic 300x300 ROI cutting window and motion trajectory.

Controls:
  - [SPACE]     : Play / Pause playback
  - [S] / [W]   : Step forward / backward 1 frame
  - [D] / [A]   : Jump forward / backward 15 frames (~0.25s)
  - [N] / [P]   : Next / Previous video
  - [1] - [9]   : Set playback speed (1=Slow 10 FPS, 5=Realtime 60 FPS, 9=Max)
  - [E]         : Export current frame overlay snapshot to `results/tracking_snapshots/`
  - [Q] / [ESC] : Quit player
=============================================================
"""

import sys
import os
import glob
import time
import cv2
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    RAW_VIDEOS_DIR,
    EXTRACTED_FRAMES_DIR,
    TRACKING_RESULTS_PATH,
    RESULTS_DIR,
    sanitize_filename,
    draw_tool_tip_blue_dot
)


def load_tracking_data(csv_path: str = TRACKING_RESULTS_PATH) -> pd.DataFrame:
    """Loads master tracking trajectory dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Tracking results CSV not found at '{csv_path}'. Run scripts/run_phase1_tracker.py first.")
    return pd.read_csv(csv_path)


def run_tracking_overlay_player():
    """Interactive visual player rendering blue dot tool tip tracking results."""
    df_tracking = load_tracking_data()
    video_names = df_tracking["video"].unique().tolist()

    if not video_names:
        print("[!] No tracking records found in CSV.")
        return

    window_name = "Phase 1: CSRT Tool Tip Tracking Player (Blue Dot Overlay)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    fps_delays = {
        '1': 100,  # 10 FPS
        '2': 50,   # 20 FPS
        '3': 33,   # 30 FPS
        '4': 20,   # 50 FPS
        '5': 16,   # ~60 FPS
        '6': 10,   # 100 FPS
        '7': 5,    # 200 FPS
        '8': 2,    # 500 FPS
        '9': 1     # Max speed
    }
    current_delay = 16
    current_vid_idx = 0

    snapshot_dir = os.path.join(RESULTS_DIR, "tracking_snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    while True:
        v_name = video_names[current_vid_idx]
        clean_name = sanitize_filename(v_name)
        v_df = df_tracking[df_tracking["video"] == v_name].sort_values("frame_idx").reset_index(drop=True)

        if len(v_df) == 0:
            current_vid_idx = (current_vid_idx + 1) % len(video_names)
            continue

        v_idx = int(v_df.iloc[0].get("video_index", current_vid_idx + 1))
        material = str(v_df.iloc[0].get("material", "Unknown"))
        total_frames = len(v_df)

        # Discover extracted frames folder or video file
        extracted_dir = os.path.join(EXTRACTED_FRAMES_DIR, clean_name)
        frame_files = sorted(glob.glob(os.path.join(extracted_dir, "frame_*.jpg")))
        is_frame_source = len(frame_files) > 0
        cap = None

        if not is_frame_source:
            raw_path = os.path.join(RAW_VIDEOS_DIR, v_name)
            cap = cv2.VideoCapture(raw_path)

        print("\n" + "=" * 75)
        print(f"  PLAYING TRACKED VIDEO [{current_vid_idx + 1}/{len(video_names)}]: Video {v_idx:02d} ({material})")
        print(f"  Filename: {v_name} | Frames: {total_frames:,}")
        print("  Controls: [SPACE] Pause | [S]/[W] Step | [A]/[D] Jump 15 | [N]/[P] Video | [E] Save Snapshot | [Q] Quit")
        print("=" * 75)

        is_paused = False
        cur_f_idx = 0
        trail_points = []

        # Create quick lookup by frame_idx
        v_lookup = {row["frame_idx"]: row for row in v_df.to_dict("records")}

        while cur_f_idx < total_frames:
            # Read Frame
            if is_frame_source:
                if cur_f_idx < len(frame_files):
                    frame = cv2.imread(frame_files[cur_f_idx])
                else:
                    frame = None
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, cur_f_idx)
                ret, frame = cap.read()
                if not ret:
                    frame = None

            if frame is None:
                break

            display = frame.copy()
            H, W = display.shape[:2]

            row = v_lookup.get(cur_f_idx)
            if row is not None:
                tx = int(row["tool_tip_x"])
                ty = int(row["tool_tip_y"])
                bx = int(row.get("bbox_x", tx - 10))
                by = int(row.get("bbox_y", ty - 10))
                bw = int(row.get("bbox_w", 20))
                bh = int(row.get("bbox_h", 20))
                rx1 = int(row.get("roi_x1", max(0, tx - 150)))
                ry1 = int(row.get("roi_y1", max(0, ty - 150)))
                rx2 = int(row.get("roi_x2", min(W, tx + 150)))
                ry2 = int(row.get("roi_y2", min(H, ty + 150)))
                status = str(row.get("tracking_status", "TRACKED"))

                # --- 1. Draw Dynamic 300x300 ROI Window (Yellow / Gold) ---
                cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (0, 220, 255), 2, lineType=cv2.LINE_AA)
                cv2.putText(display, "300x300 ROI", (rx1 + 6, ry1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

                # --- 2. Draw CSRT Bounding Box (Cyan) ---
                cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (255, 255, 0), 1, lineType=cv2.LINE_AA)

                # --- 3. Tool Tip Blue Dot removed per user preference ---

                # --- 4. Picture-in-Picture (PiP) Zoom in Top-Right ---
                pip_size = 190
                crop_patch = frame[ry1:ry2, rx1:rx2]
                if crop_patch.size > 0:
                    pip_zoom = cv2.resize(crop_patch, (pip_size, pip_size))
                    # Draw blue dot inside PiP zoom
                    scale_x = pip_size / max(1, (rx2 - rx1))
                    scale_y = pip_size / max(1, (ry2 - ry1))
                    pip_tx = int((tx - rx1) * scale_x)
                    pip_ty = int((ty - ry1) * scale_y)
                    draw_tool_tip_blue_dot(pip_zoom, pip_tx, pip_ty, radius=6, with_label=False)
                    cv2.rectangle(pip_zoom, (0, 0), (pip_size - 1, pip_size - 1), (0, 220, 255), 2)
                    cv2.putText(pip_zoom, "ROI ZOOM (300x300)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1)
                    display[12:12 + pip_size, W - pip_size - 12:W - 12] = pip_zoom

            # --- Top Info Banner ---
            cv2.rectangle(display, (0, 0), (W, 45), (15, 15, 15), -1)
            header = (
                f"Video {v_idx:02d}/27: {material:<10s} | Frame: {cur_f_idx + 1:,}/{total_frames:,} "
                f"({(cur_f_idx + 1)/total_frames*100:.1f}%) | {'PAUSED' if is_paused else 'PLAYING'}"
            )
            cv2.putText(display, header, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)

            # --- Bottom Controls Bar ---
            cv2.rectangle(display, (0, H - 45), (W, H), (15, 15, 15), -1)
            bar_text = "[SPACE] Play/Pause | [S]/[W] Step 1 | [A]/[D] +/-15 | [N]/[P] Video | [E] Snapshot | [1-9] Speed | [Q] Quit"
            cv2.putText(display, bar_text, (15, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1)

            cv2.imshow(window_name, display)

            wait_t = 0 if is_paused else current_delay
            key = cv2.waitKey(max(1, wait_t)) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                if cap is not None:
                    cap.release()
                cv2.destroyAllWindows()
                print("\n[+] Closed Tracking Player.")
                return

            elif key == ord(' '):  # SPACE
                is_paused = not is_paused

            elif key == ord('s'):  # S (+1 frame)
                is_paused = True
                cur_f_idx = min(total_frames - 1, cur_f_idx + 1)

            elif key == ord('w'):  # W (-1 frame)
                is_paused = True
                cur_f_idx = max(0, cur_f_idx - 1)

            elif key == ord('d'):  # D (+15 frames)
                cur_f_idx = min(total_frames - 1, cur_f_idx + 15)

            elif key == ord('a'):  # A (-15 frames)
                cur_f_idx = max(0, cur_f_idx - 15)

            elif key == ord('e'):  # E (Export snapshot)
                out_snap = os.path.join(snapshot_dir, f"{clean_name}_frame{cur_f_idx:06d}_overlay.jpg")
                cv2.imwrite(out_snap, display)
                print(f"  [+] Saved snapshot to: '{out_snap}'")

            elif key == ord('n'):  # N (Next video)
                current_vid_idx = (current_vid_idx + 1) % len(video_names)
                break

            elif key == ord('p'):  # P (Previous video)
                current_vid_idx = (current_vid_idx - 1 + len(video_names)) % len(video_names)
                break

            elif chr(key) in fps_delays:
                current_delay = fps_delays[chr(key)]
                print(f"  Speed set to: Key {chr(key)}")

            if not is_paused:
                cur_f_idx += 1

        if cap is not None:
            cap.release()

        if cur_f_idx >= total_frames and key not in [ord('n'), ord('p')]:
            current_vid_idx = (current_vid_idx + 1) % len(video_names)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_tracking_overlay_player()
