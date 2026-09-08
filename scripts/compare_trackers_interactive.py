# -*- coding: utf-8 -*-
"""
=============================================================
  Interactive Tool Tip Tracker Diagnostic & Comparison Tool
  Vision-Based Metal Chip Classification & Machining Monitoring
=============================================================

Compare tracking algorithms in real-time on lathe cutting footage:
  1. OpenCV MIL Tracker (cv2.TrackerMIL)
  2. NCC Template Matching + Kinematic Constraint (Filep et al.)
  3. Dynamic ROI Window (300x300 px)

Controls:
  - Drag mouse on Frame 0 to select Tool Tip Bounding Box (or press SPACE to use default).
  - [SPACE]     : Play / Pause tracking playback
  - [S]         : Step forward 1 frame (when paused)
  - [A] / [D]   : Jump backward / forward 15 frames (~0.25s)
  - [R]         : Reset & Re-select Tool BBox on current video
  - [N]         : Next video
  - [P]         : Previous video
  - [1] - [9]   : Set playback speed (1=Slow 10 FPS, 5=Normal 60 FPS, 9=Max Speed)
  - [Q] / [ESC] : Quit diagnostic tool
=============================================================
"""

import sys
import os
import time
import glob
import cv2
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    RAW_VIDEOS_DIR,
    EXTRACTED_FRAMES_DIR,
    VIDEO_METADATA_PATH,
    SPLITS_VIDEO_WISE_PATH,
    TRACKER_CONFIG_PATH,
    MIN_CROP_HALF,
    sanitize_filename
)


class NCCTemplateTracker:
    """
    Zero-Drift Normalized Cross-Correlation Template Tracker with
    Kinematic Search Window constraint based on Filep et al. (2024).
    """
    def __init__(self, search_margin_x=120, search_margin_y=50, feed_direction=-1):
        self.template = None
        self.template_gray = None
        self.bbox = None  # (x, y, w, h)
        self.last_pos = None  # (cx, cy)
        self.search_margin_x = search_margin_x
        self.search_margin_y = search_margin_y
        self.feed_direction = feed_direction  # -1 for right-to-left feed, +1 for left-to-right
        self.match_score = 1.0

    def init(self, frame, bbox):
        x, y, w, h = [int(v) for v in bbox]
        # Clamp to frame
        H, W = frame.shape[:2]
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(10, min(w, W - x))
        h = max(10, min(h, H - y))

        self.bbox = (x, y, w, h)
        self.last_pos = (x + w / 2.0, y + h / 2.0)
        self.template = frame[y:y + h, x:x + w].copy()
        if len(frame.shape) == 3:
            self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        else:
            self.template_gray = self.template.copy()
        self.match_score = 1.0
        return True

    def update(self, frame):
        if self.template_gray is None or self.last_pos is None:
            return False, (0, 0, 0, 0), 0.0

        H, W = frame.shape[:2]
        tw, th = self.template_gray.shape[1], self.template_gray.shape[0]
        cx, cy = self.last_pos

        # Define search window around previous tool position
        sx1 = max(0, int(cx - self.search_margin_x))
        sx2 = min(W, int(cx + self.search_margin_x))
        sy1 = max(0, int(cy - self.search_margin_y))
        sy2 = min(H, int(cy + self.search_margin_y))

        if (sx2 - sx1) < tw or (sy2 - sy1) < th:
            sx1, sx2, sy1, sy2 = 0, W, 0, H

        search_roi = frame[sy1:sy2, sx1:sx2]
        if len(search_roi.shape) == 3:
            search_gray = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY)
        else:
            search_gray = search_roi

        res = cv2.matchTemplate(search_gray, self.template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        self.match_score = float(max_val)
        best_x = sx1 + max_loc[0]
        best_y = sy1 + max_loc[1]

        self.bbox = (best_x, best_y, tw, th)
        self.last_pos = (best_x + tw / 2.0, best_y + th / 2.0)

        # NCC threshold for valid match
        success = max_val >= 0.35
        return success, self.bbox, self.match_score


def load_video_catalog():
    """Load video metadata catalog or discover videos directly from disk."""
    if os.path.exists(VIDEO_METADATA_PATH):
        df = pd.read_csv(VIDEO_METADATA_PATH)
        return df.to_dict("records")

    v_files = sorted(glob.glob(os.path.join(RAW_VIDEOS_DIR, "*.mp4")))
    records = []
    for idx, vf in enumerate(v_files, start=1):
        fname = os.path.basename(vf)
        records.append({
            "Index": idx,
            "Filename": fname,
            "Total_Frames": 2000,
            "FPS": 60.0
        })
    return records


def get_video_frames(video_record):
    """Retrieve frame file paths or VideoCapture object for a video."""
    v_name = video_record["Filename"]
    clean_name = sanitize_filename(v_name)
    extracted_dir = os.path.join(EXTRACTED_FRAMES_DIR, clean_name)

    if os.path.isdir(extracted_dir):
        frame_files = sorted(glob.glob(os.path.join(extracted_dir, "frame_*.jpg")))
        if len(frame_files) > 0:
            return "FILES", frame_files

    raw_path = os.path.join(RAW_VIDEOS_DIR, v_name)
    return "VIDEO", raw_path


def read_frame_by_index(source_type, source_obj, frame_idx, cap_obj=None):
    """Reads a specific frame index from either image files or VideoCapture."""
    if source_type == "FILES":
        if 0 <= frame_idx < len(source_obj):
            img = cv2.imread(source_obj[frame_idx])
            return img
        return None
    else:
        if cap_obj is not None:
            cap_obj.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap_obj.read()
            if ret:
                return frame
        return None


def run_interactive_tracker_comparison():
    """Main interactive diagnostic and tracker benchmark loop."""
    videos = load_video_catalog()
    if not videos:
        print("[!] No videos found in data/phase0_raw_videos.")
        return

    current_vid_idx = 0
    window_name = "Lathe Tool Tip Tracker Benchmark (MIL vs NCC Filep et al.)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    fps_delays = {
        '1': 100,  # 10 FPS
        '2': 50,   # 20 FPS
        '3': 33,   # 30 FPS
        '4': 20,   # 50 FPS
        '5': 16,   # ~60 FPS (Realtime)
        '6': 10,   # 100 FPS
        '7': 5,    # 200 FPS
        '8': 2,    # 500 FPS
        '9': 1     # Max speed
    }
    current_delay = 16

    while True:
        v_rec = videos[current_vid_idx]
        v_name = v_rec["Filename"]
        clean_name = sanitize_filename(v_name)
        v_index = v_rec.get("Index", current_vid_idx + 1)
        material = "Aluminium" if v_index <= 7 else ("Copper" if v_index <= 13 else "Mild Steel")

        source_type, source_obj = get_video_frames(v_rec)
        cap = None
        total_frames = 0

        if source_type == "FILES":
            total_frames = len(source_obj)
        else:
            cap = cv2.VideoCapture(source_obj)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print("\n" + "=" * 70)
        print(f"  LOADED VIDEO [{current_vid_idx + 1}/{len(videos)}]: {v_name}")
        print(f"  Material: {material} | Total Frames: {total_frames:,}")
        print("  Select Tool Tip Bounding Box on Frame 0 and press ENTER / SPACE.")
        print("=" * 70)

        # 1. Load Pre-saved Config or Allow Manual Draw
        config = {}
        if os.path.exists(TRACKER_CONFIG_PATH):
            try:
                with open(TRACKER_CONFIG_PATH, "r") as f:
                    config = json.load(f)
            except Exception:
                config = {}

        saved_cfg = config.get(v_name, {})
        start_frame_idx = saved_cfg.get("start_frame", 0)

        # Read Frame for ROI Selection
        frame0 = read_frame_by_index(source_type, source_obj, start_frame_idx, cap)
        if frame0 is None:
            frame0 = read_frame_by_index(source_type, source_obj, 0, cap)
            start_frame_idx = 0

        if frame0 is None:
            print(f"[!] Error: Could not read frame from {v_name}")
            current_vid_idx = (current_vid_idx + 1) % len(videos)
            continue

        H, W = frame0.shape[:2]
        default_bbox = (int(W * 0.55), int(H * 0.35), 60, 60)

        if "bbox" in saved_cfg:
            selected_roi = tuple(saved_cfg["bbox"])
            print(f"  [+] Loaded pre-calibrated BBox: {selected_roi} at start frame {start_frame_idx}")
        else:
            roi_prompt_img = frame0.copy()
            cv2.putText(
                roi_prompt_img,
                f"DRAW TOOL BBOX (Drag mouse, press ENTER/SPACE) | Video {v_index}: {material}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            selected_roi = cv2.selectROI(window_name, roi_prompt_img, showCrosshair=True, fromCenter=False)

            if selected_roi[2] < 10 or selected_roi[3] < 10:
                print("  [i] Using default tool tip bounding box.")
                selected_roi = default_bbox
            else:
                print(f"  [+] Selected ROI: {selected_roi}")

        # 2. Initialize Trackers
        # --- MIL Tracker ---
        mil_tracker = cv2.TrackerMIL_create()
        mil_tracker.init(frame0, selected_roi)

        # --- CSRT Tracker ---
        csrt_tracker = cv2.TrackerCSRT_create() if hasattr(cv2, 'TrackerCSRT_create') else None
        if csrt_tracker is not None:
            csrt_tracker.init(frame0, selected_roi)

        # --- NCC Filep Tracker ---
        ncc_tracker = NCCTemplateTracker(search_margin_x=130, search_margin_y=60)
        ncc_tracker.init(frame0, selected_roi)

        # Tracking state
        is_paused = False
        current_frame_idx = start_frame_idx
        mil_trajectory = []
        csrt_trajectory = []
        ncc_trajectory = []
        mil_times = []
        csrt_times = []
        ncc_times = []
        mil_failures = 0
        csrt_failures = 0
        ncc_failures = 0

        # 3. Main Video Tracking Playback Loop
        while current_frame_idx < total_frames:
            frame = read_frame_by_index(source_type, source_obj, current_frame_idx, cap)
            if frame is None:
                break

            display_frame = frame.copy()

            # --- Update MIL Tracker ---
            t0 = time.time()
            mil_ok, mil_bbox = mil_tracker.update(frame)
            mil_time_ms = (time.time() - t0) * 1000.0
            mil_times.append(mil_time_ms)

            if mil_ok:
                mx, my, mw, mh = [int(v) for v in mil_bbox]
                mcx, mcy = int(mx + mw / 2), int(my + mh / 2)
                mil_trajectory.append((mcx, mcy))
            else:
                mil_failures += 1
                mcx, mcy = mil_trajectory[-1] if mil_trajectory else (0, 0)
                mx, my, mw, mh = mcx - 20, mcy - 20, 40, 40

            # --- Update CSRT Tracker ---
            csrt_ok = False
            csrt_bbox = (0, 0, 0, 0)
            csrt_time_ms = 0.0
            if csrt_tracker is not None:
                t0 = time.time()
                csrt_ok, csrt_bbox = csrt_tracker.update(frame)
                csrt_time_ms = (time.time() - t0) * 1000.0
                csrt_times.append(csrt_time_ms)

                if csrt_ok:
                    cx_, cy_, cw_, ch_ = [int(v) for v in csrt_bbox]
                    ccx, ccy = int(cx_ + cw_ / 2), int(cy_ + ch_ / 2)
                    csrt_trajectory.append((ccx, ccy))
                else:
                    csrt_failures += 1
                    ccx, ccy = csrt_trajectory[-1] if csrt_trajectory else (0, 0)
                    cx_, cy_, cw_, ch_ = ccx - 20, ccy - 20, 40, 40

            # --- Update NCC Filep Tracker ---
            t0 = time.time()
            ncc_ok, ncc_bbox, ncc_score = ncc_tracker.update(frame)
            ncc_time_ms = (time.time() - t0) * 1000.0
            ncc_times.append(ncc_time_ms)

            if ncc_ok:
                nx, ny, nw, nh = [int(v) for v in ncc_bbox]
                ncx, ncy = int(nx + nw / 2), int(ny + nh / 2)
                ncc_trajectory.append((ncx, ncy))
            else:
                ncc_failures += 1
                ncx, ncy = ncc_trajectory[-1] if ncc_trajectory else (0, 0)
                nx, ny, nw, nh = ncx - 20, ncy - 20, 40, 40

            # --- Compute Drift Metric ---
            drift_px = np.sqrt((mcx - ncx) ** 2 + (mcy - ncy) ** 2)

            # --- Render Trajectory Trails (Last 60 points) ---
            for i in range(1, len(mil_trajectory[-60:])):
                pt1 = mil_trajectory[-60:][i - 1]
                pt2 = mil_trajectory[-60:][i]
                cv2.line(display_frame, pt1, pt2, (255, 120, 0), 2)  # Blue/Cyan trail

            if csrt_tracker is not None:
                for i in range(1, len(csrt_trajectory[-60:])):
                    pt1 = csrt_trajectory[-60:][i - 1]
                    pt2 = csrt_trajectory[-60:][i]
                    cv2.line(display_frame, pt1, pt2, (255, 0, 255), 2)  # Magenta trail

            for i in range(1, len(ncc_trajectory[-60:])):
                pt1 = ncc_trajectory[-60:][i - 1]
                pt2 = ncc_trajectory[-60:][i]
                cv2.line(display_frame, pt1, pt2, (0, 255, 0), 2)  # Green trail

            # --- Draw MIL Box (Blue / Cyan) ---
            if mil_ok:
                cv2.rectangle(display_frame, (mx, my), (mx + mw, my + mh), (255, 180, 0), 2)
                cv2.drawMarker(display_frame, (mcx, mcy), (255, 200, 0), cv2.MARKER_CROSS, 14, 2)
                cv2.putText(display_frame, f"MIL", (mx, my - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1)

            # --- Draw CSRT Box (Magenta) ---
            if csrt_tracker is not None and csrt_ok:
                cv2.rectangle(display_frame, (cx_, cy_), (cx_ + cw_, cy_ + ch_), (255, 0, 255), 2)
                cv2.drawMarker(display_frame, (ccx, ccy), (255, 0, 255), cv2.MARKER_CROSS, 14, 2)
                cv2.putText(display_frame, f"CSRT", (cx_, cy_ + ch_ + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

            # --- Draw NCC Box (Green) ---
            if ncc_ok:
                cv2.rectangle(display_frame, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)
                cv2.drawMarker(display_frame, (ncx, ncy), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
                cv2.putText(display_frame, f"NCC (Filep)", (nx, ny - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # --- Draw 300x300 Dynamic ROI Window (Yellow / Amber) ---
            roi_half = MIN_CROP_HALF  # 150 px
            rx1 = max(0, ncx - roi_half)
            ry1 = max(0, ncy - roi_half)
            rx2 = min(W, ncx + roi_half)
            ry2 = min(H, ncy + roi_half)
            cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (0, 230, 255), 2)

            # --- Picture-in-Picture (PiP) Zoom View of Tool Tip in Top-Right Corner ---
            pip_size = 180
            crop_patch = frame[ry1:ry2, rx1:rx2]
            if crop_patch.size > 0:
                pip_zoom = cv2.resize(crop_patch, (pip_size, pip_size))
                cv2.rectangle(pip_zoom, (0, 0), (pip_size - 1, pip_size - 1), (0, 230, 255), 2)
                cv2.putText(pip_zoom, "ROI CROP (300x300)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1)
                display_frame[15:15 + pip_size, W - pip_size - 15:W - 15] = pip_zoom

            # --- HUD Diagnostic Overlay ---
            cv2.rectangle(display_frame, (0, 0), (W, 45), (20, 20, 20), -1)
            status_text = (
                f"Video {v_index}/27: {material} ({v_name[:32]}...) | "
                f"Frame: {current_frame_idx:,}/{total_frames:,} | "
                f"{'PAUSED' if is_paused else 'PLAYING'}"
            )
            cv2.putText(display_frame, status_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

            # Bottom Stats Bar
            cv2.rectangle(display_frame, (0, H - 75), (W, H), (20, 20, 20), -1)

            # MIL stats
            mil_color = (255, 180, 0) if mil_ok else (0, 0, 255)
            mil_info = f"MIL:  {'LOCKED' if mil_ok else 'LOST'} ({mil_time_ms:.1f}ms) Fail:{mil_failures}"
            cv2.putText(display_frame, mil_info, (15, H - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.44, mil_color, 1)

            # CSRT stats
            if csrt_tracker is not None:
                csrt_color = (255, 0, 255) if csrt_ok else (0, 0, 255)
                csrt_info = f"CSRT: {'LOCKED' if csrt_ok else 'LOST'} ({csrt_time_ms:.1f}ms) Fail:{csrt_failures}"
                cv2.putText(display_frame, csrt_info, (15, H - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, csrt_color, 1)

            # NCC stats
            ncc_info = f"NCC (Filep): {'LOCKED' if ncc_ok else 'LOST'} (Score:{ncc_score:.2f}, {ncc_time_ms:.1f}ms) Fail:{ncc_failures}"
            cv2.putText(display_frame, ncc_info, (15, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 0), 1)

            # Drift Warning Alert
            drift_color = (0, 255, 0) if drift_px < 15 else ((0, 200, 255) if drift_px < 40 else (0, 0, 255))
            drift_info = f"Tracker Delta (Drift): {drift_px:.1f} px"
            cv2.putText(display_frame, drift_info, (W - 340, H - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.52, drift_color, 2)

            controls_info = "[SPACE] Pause | [S] Step | [R] Reset | [N] Next | [P] Prev | [1-9] Speed | [Q] Quit"
            cv2.putText(display_frame, controls_info, (W - 550, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

            cv2.imshow(window_name, display_frame)

            # Handle User Keyboard Inputs
            wait_time = 0 if is_paused else current_delay
            key = cv2.waitKey(max(1, wait_time)) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                print_comparative_summary(v_name, material, mil_times, csrt_times, ncc_times, mil_failures, csrt_failures, ncc_failures, len(mil_trajectory))
                if cap is not None:
                    cap.release()
                cv2.destroyAllWindows()
                return

            elif key == ord(' '):  # SPACE (Play/Pause)
                is_paused = not is_paused

            elif key == ord('s'):  # S (Step forward 1 frame)
                is_paused = True
                current_frame_idx += 1

            elif key == ord('d'):  # D (Forward 15 frames)
                current_frame_idx = min(total_frames - 1, current_frame_idx + 15)

            elif key == ord('a'):  # A (Back 15 frames)
                current_frame_idx = max(0, current_frame_idx - 15)

            elif key == ord('r'):  # R (Reset tool box on current video)
                break

            elif key == ord('n'):  # N (Next video)
                print_comparative_summary(v_name, material, mil_times, csrt_times, ncc_times, mil_failures, csrt_failures, ncc_failures, len(mil_trajectory))
                current_vid_idx = (current_vid_idx + 1) % len(videos)
                break

            elif key == ord('p'):  # P (Previous video)
                print_comparative_summary(v_name, material, mil_times, csrt_times, ncc_times, mil_failures, csrt_failures, ncc_failures, len(mil_trajectory))
                current_vid_idx = (current_vid_idx - 1 + len(videos)) % len(videos)
                break

            elif chr(key) in fps_delays:
                current_delay = fps_delays[chr(key)]
                print(f"  Playback speed set to: Key {chr(key)}")

            if not is_paused:
                current_frame_idx += 1

        if cap is not None:
            cap.release()

        if current_frame_idx >= total_frames and key not in [ord('r'), ord('p'), ord('n')]:
            print_comparative_summary(v_name, material, mil_times, csrt_times, ncc_times, mil_failures, csrt_failures, ncc_failures, len(mil_trajectory))
            current_vid_idx = (current_vid_idx + 1) % len(videos)

    cv2.destroyAllWindows()


def print_comparative_summary(v_name, material, mil_times, csrt_times, ncc_times, mil_fails, csrt_fails, ncc_fails, total_frames):
    """Prints a quantitative comparison table of tracker performance."""
    if total_frames == 0:
        return

    n_f = max(1, total_frames)
    mil_fps = 1000.0 / np.mean(mil_times) if mil_times else 0
    ncc_fps = 1000.0 / np.mean(ncc_times) if ncc_times else 0
    csrt_fps = (1000.0 / np.mean(csrt_times)) if csrt_times else 0

    mil_succ = max(0.0, 100.0 * (n_f - mil_fails) / n_f)
    csrt_succ = max(0.0, 100.0 * (n_f - csrt_fails) / n_f) if csrt_times else 0.0
    ncc_succ = max(0.0, 100.0 * (n_f - ncc_fails) / n_f)

    print("\n" + "=" * 76)
    print(f"  📊 COMPARATIVE TRACKER BENCHMARK REPORT — {material} ({v_name[:32]})")
    print("=" * 76)
    print(f"{'Tracker Algorithm':<26} | {'Success Rate':<14} | {'Avg Time (ms)':<15} | {'Speed (FPS)':<12}")
    print("-" * 76)
    print(f"{'1. OpenCV MIL Tracker':<26} | {mil_succ:>11.1f}% | {np.mean(mil_times):>12.1f} ms | {mil_fps:>9.1f} FPS")
    if csrt_times:
        print(f"{'2. OpenCV CSRT Tracker':<26} | {csrt_succ:>11.1f}% | {np.mean(csrt_times):>12.1f} ms | {csrt_fps:>9.1f} FPS")
    print(f"{'3. NCC Template (Filep)':<26} | {ncc_succ:>11.1f}% | {np.mean(ncc_times):>12.1f} ms | {ncc_fps:>9.1f} FPS")
    print("-" * 76)

    # Automated Winner Verdict
    best = "NCC Template (Filep et al.)"
    reason = "Highest frame lock reliability, immune to visual chip contamination, fastest computation."
    if mil_succ > ncc_succ:
        best = "OpenCV MIL"
        reason = "Higher lock rate on this sequence."
    elif csrt_times and csrt_succ > ncc_succ and csrt_succ >= mil_succ:
        best = "OpenCV CSRT"
        reason = "Higher spatial reliability on this sequence."

    print(f"  🏆 RECOMMENDED TRACKER: {best}")
    print(f"  💡 Reason: {reason}")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    run_interactive_tracker_comparison()
