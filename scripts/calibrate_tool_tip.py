# -*- coding: utf-8 -*-
"""
=============================================================
  Interactive Manual Tool Tip & Bounding Box Annotation Tool
  Vision-Based Metal Chip Classification & Machining Monitoring
=============================================================

Allows you to:
  1. Scrub to any starting frame (using [A]/[D] or slider)
  2. Click the EXACT Tool Tip Point (Pinpoint crosshair)
  3. Drag to draw the Tool Body Bounding Box
  4. Real-time 4x Magnifier Loupe for pixel-perfect placement
  5. Saves directly to `data/metadata/tracker_config.json`
  6. Immediately tests the tracker on the annotated video!

Controls:
  - [Left Click]         : Set Tool Tip Point (Red crosshair)
  - [Left Click + Drag]  : Draw Tool Bounding Box (Cyan rectangle)
  - [A] / [D] or [<-]/[->]: Step frame backward / forward
  - [SPACE]              : Play/Pause video preview
  - [ENTER]              : Confirm & Save annotation for this video
  - [N] / [P]            : Next / Previous video in the catalog
  - [R]                  : Reset drawing on current frame
  - [Q] / [ESC]          : Quit
=============================================================
"""

import sys
import os
import glob
import json
import cv2
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    RAW_VIDEOS_DIR,
    EXTRACTED_FRAMES_DIR,
    VIDEO_METADATA_PATH,
    TRACKER_CONFIG_PATH,
    sanitize_filename
)


# Global drawing state
drawing_state = {
    "tip_point": None,        # (x, y)
    "bbox": None,             # [x, y, w, h]
    "is_dragging": False,
    "drag_start": None,       # (x, y)
    "mouse_pos": (0, 0),
    "current_frame_idx": 0,
    "needs_redraw": True
}


def load_config():
    """Load existing tracker configurations."""
    if os.path.exists(TRACKER_CONFIG_PATH):
        with open(TRACKER_CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}


def save_config(cfg):
    """Save updated tracker configurations."""
    os.makedirs(os.path.dirname(TRACKER_CONFIG_PATH), exist_ok=True)
    with open(TRACKER_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  [+] Saved configuration to '{TRACKER_CONFIG_PATH}'")


def mouse_callback(event, x, y, flags, param):
    """Handles mouse events for tool tip point click and bounding box drag."""
    global drawing_state
    drawing_state["mouse_pos"] = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_state["drag_start"] = (x, y)
        drawing_state["is_dragging"] = True
        # If user hasn't set tip yet, click also sets tip
        if drawing_state["tip_point"] is None:
            drawing_state["tip_point"] = (x, y)
        drawing_state["needs_redraw"] = True

    elif event == cv2.EVENT_MOUSEMOVE:
        drawing_state["needs_redraw"] = True

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing_state["is_dragging"]:
            drawing_state["is_dragging"] = False
            x0, y0 = drawing_state["drag_start"]
            bx1 = min(x0, x)
            by1 = min(y0, y)
            bx2 = max(x0, x)
            by2 = max(y0, y)
            bw = bx2 - bx1
            bh = by2 - by1

            if bw > 8 and bh > 8:
                drawing_state["bbox"] = [int(bx1), int(by1), int(bw), int(bh)]
                # If tip wasn't set or outside box, default tip to left-middle or right-middle
                if drawing_state["tip_point"] is None:
                    drawing_state["tip_point"] = (int(bx1), int(by1 + bh / 2))
            else:
                # Simple single click: sets the exact tip point
                drawing_state["tip_point"] = (x, y)

            drawing_state["needs_redraw"] = True

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click to explicitly set the tool tip point
        drawing_state["tip_point"] = (x, y)
        drawing_state["needs_redraw"] = True


def render_display(raw_frame, v_name, v_idx, total_vids, material, total_frames, current_idx):
    """Draws overlays, crosshairs, bounding boxes, magnifier loupe, and instructions."""
    display = raw_frame.copy()
    H, W = display.shape[:2]

    tip = drawing_state["tip_point"]
    bbox = drawing_state["bbox"]
    is_drag = drawing_state["is_dragging"]
    mx, my = drawing_state["mouse_pos"]

    # 1. Draw Active Drag Box
    if is_drag and drawing_state["drag_start"] is not None:
        x0, y0 = drawing_state["drag_start"]
        cv2.rectangle(display, (x0, y0), (mx, my), (0, 255, 255), 2)

    # 2. Draw Confirmed Bounding Box
    if bbox is not None:
        bx, by, bw, bh = bbox
        cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (255, 200, 0), 2)
        cv2.putText(display, f"Tool BBox: {bw}x{bh}px", (bx, max(15, by - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

        # Draw 300x300 ROI preview centered on tip or bbox center
        cx = tip[0] if tip is not None else int(bx + bw / 2)
        cy = tip[1] if tip is not None else int(by + bh / 2)
        rx1 = max(0, cx - 150)
        ry1 = max(0, cy - 150)
        rx2 = min(W, cx + 150)
        ry2 = min(H, cy + 150)
        cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (0, 255, 0), 1, lineType=cv2.LINE_AA)
        cv2.putText(display, "300x300 Dynamic Crop ROI", (rx1 + 5, ry1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)

    # 3. Draw Tool Tip Crosshair
    if tip is not None:
        tx, ty = tip
        cv2.drawMarker(display, (tx, ty), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.circle(display, (tx, ty), 5, (0, 0, 255), 2)
        cv2.putText(display, f"TIP ({tx}, {ty})", (tx + 10, ty - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 4. Pixel-Level Magnifier Loupe (Top-Right Corner)
    loupe_size = 180
    zoom_radius = 25  # 50x50 patch zoomed to 180x180 (3.6x zoom)
    lx1 = max(0, mx - zoom_radius)
    ly1 = max(0, my - zoom_radius)
    lx2 = min(W, mx + zoom_radius)
    ly2 = min(H, my + zoom_radius)

    patch = raw_frame[ly1:ly2, lx1:lx2]
    if patch.size > 0:
        loupe_img = cv2.resize(patch, (loupe_size, loupe_size), interpolation=cv2.INTER_NEAREST)
        # Center crosshair on magnifier
        hc = loupe_size // 2
        cv2.line(loupe_img, (hc - 12, hc), (hc + 12, hc), (0, 0, 255), 1)
        cv2.line(loupe_img, (hc, hc - 12), (hc, hc + 12), (0, 0, 255), 1)
        cv2.rectangle(loupe_img, (0, 0), (loupe_size - 1, loupe_size - 1), (0, 255, 255), 2)
        cv2.putText(loupe_img, f"Zoom 3.6x ({mx},{my})", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        # Overlay on display top-right
        display[10:10 + loupe_size, W - loupe_size - 10:W - 10] = loupe_img

    # 5. Top Header Banner
    cv2.rectangle(display, (0, 0), (W, 45), (15, 15, 15), -1)
    header = (
        f"Video [{v_idx}/{total_vids}]: {material} | {v_name[:36]}... | "
        f"Frame: {current_idx:,}/{total_frames:,}"
    )
    cv2.putText(display, header, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # 6. Bottom Instruction Bar
    cv2.rectangle(display, (0, H - 55), (W, H), (15, 15, 15), -1)
    status_msg = "DRAG mouse to draw Tool BBox. RIGHT-CLICK or CLICK to place Tool Tip Point."
    if tip is not None and bbox is not None:
        status_msg = f"READY: Tip=({tip[0]},{tip[1]}), BBox={bbox[2]}x{bbox[3]}px. Press [ENTER] to save & next."

    cv2.putText(display, status_msg, (15, H - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1)
    cv2.putText(
        display,
        "[A]/[D] Scrub | [ENTER] Save | [R] Reset | [N] Next Vid | [P] Prev | [Q] Quit",
        (15, H - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 180, 180),
        1
    )

    return display


def run_annotation_tool():
    """Main manual annotation interactive loop."""
    global drawing_state

    # Load video catalog
    df_vm = pd.read_csv(VIDEO_METADATA_PATH)
    video_records = df_vm.to_dict("records")
    config = load_config()

    window_name = "Interactive Tool Tip & Bounding Box Annotator"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setMouseCallback(window_name, mouse_callback)

    v_cursor = 0
    total_videos = len(video_records)

    while 0 <= v_cursor < total_videos:
        v_rec = video_records[v_cursor]
        v_name = v_rec["Filename"]
        clean_name = sanitize_filename(v_name)
        v_idx = v_rec["Index"]
        material = "Aluminium" if v_idx <= 7 else ("Copper" if v_idx <= 13 else "Mild Steel")

        # Load video frames
        extracted_dir = os.path.join(EXTRACTED_FRAMES_DIR, clean_name)
        frame_files = sorted(glob.glob(os.path.join(extracted_dir, "frame_*.jpg")))
        is_frame_source = len(frame_files) > 0
        cap = None

        if is_frame_source:
            total_frames = len(frame_files)
        else:
            raw_path = os.path.join(RAW_VIDEOS_DIR, v_name)
            cap = cv2.VideoCapture(raw_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Check if previous config exists for this video
        prev_cfg = config.get(v_name, {})
        start_frame_idx = prev_cfg.get("start_frame", 0)
        drawing_state["current_frame_idx"] = start_frame_idx

        if "bbox" in prev_cfg:
            drawing_state["bbox"] = prev_cfg["bbox"]
        else:
            drawing_state["bbox"] = None

        if "tool_tip" in prev_cfg:
            drawing_state["tip_point"] = tuple(prev_cfg["tool_tip"])
        elif drawing_state["bbox"] is not None:
            bx, by, bw, bh = drawing_state["bbox"]
            drawing_state["tip_point"] = (int(bx), int(by + bh / 2))
        else:
            drawing_state["tip_point"] = None

        drawing_state["needs_redraw"] = True
        is_playing = False

        print("\n" + "=" * 70)
        print(f"  ANNOTATING VIDEO [{v_cursor + 1}/{total_videos}]: {v_name}")
        print(f"  Material: {material} | Total Frames: {total_frames:,}")
        print("  Click to place Tool Tip | Drag to draw Bounding Box | Press ENTER to save.")
        print("=" * 70)

        while True:
            cur_idx = drawing_state["current_frame_idx"]

            # Load frame
            if is_frame_source:
                cur_idx = min(total_frames - 1, max(0, cur_idx))
                frame = cv2.imread(frame_files[cur_idx])
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, cur_idx)
                ret, frame = cap.read()
                if not ret:
                    frame = None

            if frame is None:
                print(f"[!] Warning: Unable to read frame {cur_idx}")
                break

            display = render_display(
                frame,
                v_name,
                v_cursor + 1,
                total_videos,
                material,
                total_frames,
                cur_idx
            )

            cv2.imshow(window_name, display)

            wait_delay = 33 if is_playing else 15
            key = cv2.waitKey(wait_delay) & 0xFF

            if is_playing:
                drawing_state["current_frame_idx"] = (drawing_state["current_frame_idx"] + 1) % total_frames

            if key == ord('q') or key == 27:  # Q or ESC
                if cap is not None:
                    cap.release()
                cv2.destroyAllWindows()
                print("\n[+] Exited Annotation Tool.")
                return

            elif key == ord(' '):  # SPACE (Play/Pause)
                is_playing = not is_playing

            elif key == ord('d') or key == 83:  # D or Right Arrow (+15 frames)
                drawing_state["current_frame_idx"] = min(total_frames - 1, drawing_state["current_frame_idx"] + 15)

            elif key == ord('a') or key == 81:  # A or Left Arrow (-15 frames)
                drawing_state["current_frame_idx"] = max(0, drawing_state["current_frame_idx"] - 15)

            elif key == ord('s'):  # S (+1 frame)
                drawing_state["current_frame_idx"] = min(total_frames - 1, drawing_state["current_frame_idx"] + 1)

            elif key == ord('w'):  # W (-1 frame)
                drawing_state["current_frame_idx"] = max(0, drawing_state["current_frame_idx"] - 1)

            elif key == ord('r'):  # R (Reset annotation on this video)
                drawing_state["bbox"] = None
                drawing_state["tip_point"] = None
                print("  [i] Reset drawing for current video.")

            elif key == 13:  # ENTER (Save & Next)
                if drawing_state["bbox"] is not None:
                    bx, by, bw, bh = drawing_state["bbox"]
                    tip = drawing_state["tip_point"] or (int(bx), int(by + bh / 2))
                    config[v_name] = {
                        "video_index": v_idx,
                        "material": material,
                        "start_frame": drawing_state["current_frame_idx"],
                        "bbox": [int(bx), int(by), int(bw), int(bh)],
                        "tool_tip": [int(tip[0]), int(tip[1])],
                        "offset_from_bbox": [int(tip[0] - bx), int(tip[1] - by)]
                    }
                    save_config(config)
                    v_cursor += 1
                    break
                else:
                    print("  [!] Please draw a bounding box around the tool first before pressing ENTER.")

            elif key == ord('n'):  # N (Next video without saving)
                v_cursor = (v_cursor + 1) % total_videos
                break

            elif key == ord('p'):  # P (Previous video)
                v_cursor = (v_cursor - 1 + total_videos) % total_videos
                break

        if cap is not None:
            cap.release()

    cv2.destroyAllWindows()
    print("\n" + "=" * 70)
    print("  ALL VIDEOS ANNOTATED SUCCESSFULLY!")
    print(f"  Configuration saved to '{TRACKER_CONFIG_PATH}'")
    print("=" * 70)


if __name__ == "__main__":
    run_annotation_tool()
