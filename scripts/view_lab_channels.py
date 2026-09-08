# -*- coding: utf-8 -*-
"""
Generate a side-by-side comparison image of Step 1 (LAB Channel Separation)
across Aluminium, Copper, and Mild Steel samples with correct labels.
"""

import os
import sys
import glob
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import PROJECT_ROOT, RESULTS_DIR
from src.lab_preprocessor import bgr_to_lab

os.makedirs(RESULTS_DIR, exist_ok=True)
output_img_path = str(RESULTS_DIR / "step1_lab_channels_comparison.png")

# Pick representative samples for each material
samples = [
    ("Aluminium (Vid 01)", str(PROJECT_ROOT / "data/phase2_roi_crops/retained/WhatsApp_Video_2026-08-11_at_6.55.09_AM/crop_000150.jpg")),
    ("Copper (Vid 11)", str(PROJECT_ROOT / "data/phase2_roi_crops/retained/WhatsApp_Video_2026-08-11_at_6.55.14_AM_(1)/crop_000150.jpg")),
    ("Mild Steel (Vid 16)", str(PROJECT_ROOT / "data/phase2_roi_crops/retained/WhatsApp_Video_2026-08-11_at_6.55.16_AM_(1)/crop_000150.jpg"))
]

def annotate(img, text):
    """Draws a label banner on top of an image."""
    canvas = img.copy()
    if len(canvas.shape) == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(canvas, (0, 0), (300, 32), (30, 30, 30), -1)
    cv2.putText(canvas, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas

rows = []
for material_label, path in samples:
    bgr = cv2.imread(path)
    if bgr is None:
        continue
    
    L, A, B = bgr_to_lab(bgr)

    # Colorize A and B channels for intuitive inspection
    a_color = cv2.applyColorMap(A, cv2.COLORMAP_TURBO)
    b_color = cv2.applyColorMap(B, cv2.COLORMAP_VIRIDIS)

    col1 = annotate(bgr, f"{material_label}: Raw BGR")
    col2 = annotate(L, "L (Lightness / Grayscale)")
    col3 = annotate(a_color, "A Channel (Green <-> Red)")
    col4 = annotate(b_color, "B Channel (Blue <-> Yellow)")

    row = np.hstack([col1, col2, col3, col4])
    rows.append(row)

grid = np.vstack(rows)

header = np.zeros((50, grid.shape[1], 3), dtype=np.uint8) + 20
cv2.putText(header, "STEP 1: CIELAB DECOUPLING (L = Luminance, A & B = Chrominance)", (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2, cv2.LINE_AA)

final_image = np.vstack([header, grid])
cv2.imwrite(output_img_path, final_image)
print(f"Updated comparison saved to: {output_img_path}")
