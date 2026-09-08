# -*- coding: utf-8 -*-
"""
Generate a side-by-side comparison image of Step 2 (CLAHE Lighting Equalization)
across Aluminium, Copper, and Mild Steel samples.
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import PROJECT_ROOT, RESULTS_DIR
from src.lab_preprocessor import bgr_to_lab, apply_clahe, normalize_illumination_bgr

os.makedirs(RESULTS_DIR, exist_ok=True)
output_img_path = str(RESULTS_DIR / "step2_clahe_comparison.png")

samples = [
    ("Aluminium (Vid 01)", str(PROJECT_ROOT / "data/phase2_roi_crops/retained/WhatsApp_Video_2026-08-11_at_6.55.09_AM/crop_000150.jpg")),
    ("Copper (Vid 11)", str(PROJECT_ROOT / "data/phase2_roi_crops/retained/WhatsApp_Video_2026-08-11_at_6.55.14_AM_(1)/crop_000150.jpg")),
    ("Mild Steel (Vid 16)", str(PROJECT_ROOT / "data/phase2_roi_crops/retained/WhatsApp_Video_2026-08-11_at_6.55.16_AM_(1)/crop_000150.jpg"))
]

def annotate(img, text, color=(255, 255, 255)):
    """Draws a label banner on top of an image."""
    canvas = img.copy()
    if len(canvas.shape) == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(canvas, (0, 0), (300, 32), (30, 30, 30), -1)
    cv2.putText(canvas, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return canvas

rows = []
for material_label, path in samples:
    bgr = cv2.imread(path)
    if bgr is None:
        continue
    
    L, A, B = bgr_to_lab(bgr)
    L_clahe = apply_clahe(L, clip_limit=2.5, tile_grid_size=(8, 8))
    bgr_normalized = normalize_illumination_bgr(bgr, clip_limit=2.5, tile_grid_size=(8, 8))

    col1 = annotate(bgr, f"{material_label}: Raw BGR")
    col2 = annotate(L, "1. Raw L Channel (Uneven)")
    col3 = annotate(L_clahe, "2. CLAHE L Channel (Equalized)", color=(0, 255, 255))
    col4 = annotate(bgr_normalized, "3. Normalized Color BGR", color=(0, 255, 0))

    row = np.hstack([col1, col2, col3, col4])
    rows.append(row)

grid = np.vstack(rows)

header = np.zeros((50, grid.shape[1], 3), dtype=np.uint8) + 20
cv2.putText(header, "STEP 2: SMART LOCAL LIGHTING EQUALIZATION (CLAHE on L-Channel)", (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2, cv2.LINE_AA)

final_image = np.vstack([header, grid])
cv2.imwrite(output_img_path, final_image)
print(f"Step 2 comparison saved to: {output_img_path}")
