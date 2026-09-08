# -*- coding: utf-8 -*-
"""
Generate a side-by-side comparison image of Step 3 (Bilateral Filtering / Denoising)
across Aluminium, Copper, and Mild Steel samples.
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import PROJECT_ROOT, RESULTS_DIR
from src.lab_preprocessor import get_lightness_channel, apply_clahe, apply_bilateral_filter

os.makedirs(RESULTS_DIR, exist_ok=True)
output_img_path = str(RESULTS_DIR / "step3_bilateral_comparison.png")

samples = [
    ("Aluminium (Vid 01)", str(PROJECT_ROOT / "data/phase2_roi_crops/retained/WhatsApp_Video_2026-08-11_at_6.55.09_AM/crop_000150.jpg")),
    ("Copper (Vid 11)", str(PROJECT_ROOT / "data/phase2_roi_crops/retained/WhatsApp_Video_2026-08-11_at_6.55.14_AM_(1)/crop_000150.jpg")),
    ("Mild Steel (Vid 16)", str(PROJECT_ROOT / "data/phase2_roi_crops/retained/WhatsApp_Video_2026-08-11_at_6.55.16_AM_(1)/crop_000150.jpg"))
]

def annotate(img, text, subtext="", color=(255, 255, 255)):
    """Draws a label banner on top of an image."""
    canvas = img.copy()
    if len(canvas.shape) == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(canvas, (0, 0), (300, 36), (30, 30, 30), -1)
    cv2.putText(canvas, text, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    if subtext:
        cv2.putText(canvas, subtext, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA)
    return canvas

rows = []
for material_label, path in samples:
    bgr = cv2.imread(path)
    if bgr is None:
        continue
    
    l_raw = get_lightness_channel(bgr)
    l_clahe = apply_clahe(l_raw, clip_limit=2.5, tile_grid_size=(8, 8))
    l_denoised = apply_bilateral_filter(l_clahe, d=7, sigma_color=75.0, sigma_space=75.0)

    # Compute high-frequency noise difference (what was removed)
    diff = cv2.absdiff(l_clahe, l_denoised)
    # Amplify difference for visual clarity
    diff_vis = cv2.applyColorMap(cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_INFERNO)

    col1 = annotate(bgr, f"{material_label}: Raw BGR", "Original Camera Crop")
    col2 = annotate(l_clahe, "Step 2: CLAHE L", "Has camera grain & sensor noise", color=(255, 200, 100))
    col3 = annotate(l_denoised, "Step 3: Bilateral Denoised", "Smooth surfaces + sharp chip edges", color=(0, 255, 0))
    col4 = annotate(diff_vis, "Removed Noise & Grain Map", "High-frequency speckle eliminated", color=(100, 200, 255))

    row = np.hstack([col1, col2, col3, col4])
    rows.append(row)

grid = np.vstack(rows)

header = np.zeros((50, grid.shape[1], 3), dtype=np.uint8) + 20
cv2.putText(header, "STEP 3: EDGE-PRESERVING BILATERAL DENOISING (Eliminates Grain & Speckle)", (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2, cv2.LINE_AA)

final_image = np.vstack([header, grid])
cv2.imwrite(output_img_path, final_image)
print(f"Step 3 comparison successfully saved to: {output_img_path}")
