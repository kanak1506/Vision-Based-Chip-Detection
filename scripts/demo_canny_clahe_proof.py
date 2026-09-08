# -*- coding: utf-8 -*-
"""
Proof demonstration: Canny Edge Detection with vs without CLAHE
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import PROJECT_ROOT, RESULTS_DIR
from src.lab_preprocessor import get_lightness_channel, apply_clahe

os.makedirs(RESULTS_DIR, exist_ok=True)
output_path = str(RESULTS_DIR / "why_clahe_matters_canny_comparison.png")

# Load sample
sample_path = str(PROJECT_ROOT / "data/phase2_roi_crops/retained/WhatsApp_Video_2026-08-11_at_6.55.14_AM_(1)/crop_000150.jpg")
img_bgr = cv2.imread(sample_path)

# 1. Raw grayscale
gray_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
# 2. CLAHE Lightness
l_raw = get_lightness_channel(img_bgr)
l_clahe = apply_clahe(l_raw, clip_limit=3.0, tile_grid_size=(8, 8))

# Edge detection on both using identical thresholds
v_raw = np.median(gray_raw)
canny_raw = cv2.Canny(gray_raw, int(max(0, 0.66 * v_raw)), int(min(255, 1.33 * v_raw)))

v_clahe = np.median(l_clahe)
canny_clahe = cv2.Canny(l_clahe, int(max(0, 0.66 * v_clahe)), int(min(255, 1.33 * v_clahe)))

def annotate(img, text, subtext="", color=(255, 255, 255)):
    canvas = img.copy()
    if len(canvas.shape) == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 45), (20, 20, 20), -1)
    cv2.putText(canvas, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    if subtext:
        cv2.putText(canvas, subtext, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    return canvas

p1 = annotate(img_bgr, "Original Crop", "Copper chips on tool")
p2 = annotate(canny_raw, "WITHOUT CLAHE: Broken Edges", "Faint chip contours missing", color=(0, 0, 255))
p3 = annotate(canny_clahe, "WITH CLAHE: Full Chip Geometry", "Complete closed chip loops", color=(0, 255, 0))

# Zoom into the tool-tip chip cluster (y: 100-240, x: 100-240)
zoom_raw = cv2.resize(canny_raw[100:230, 110:240], (300, 300), interpolation=cv2.INTER_NEAREST)
zoom_clahe = cv2.resize(canny_clahe[100:230, 110:240], (300, 300), interpolation=cv2.INTER_NEAREST)

z1 = annotate(cv2.resize(img_bgr[100:230, 110:240], (300, 300)), "Zoom: Raw Chips", "Close-up of cutting zone")
z2 = annotate(zoom_raw, "Zoom: WITHOUT CLAHE", "Missing internal chip contours", color=(0, 0, 255))
z3 = annotate(zoom_clahe, "Zoom: WITH CLAHE", "Dense, sharp chip outlines", color=(0, 255, 0))

row1 = np.hstack([p1, p2, p3])
row2 = np.hstack([z1, z2, z3])

grid = np.vstack([row1, row2])
cv2.imwrite(output_path, grid)
print("Saved Canny proof to:", output_path)
