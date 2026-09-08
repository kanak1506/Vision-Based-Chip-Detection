# -*- coding: utf-8 -*-
"""
Generate a side-by-side comparison image of Step 3.5 (Asymmetric Cutting Sector Masking)
across Aluminium, Copper, and Mild Steel samples.
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import PROJECT_ROOT, RESULTS_DIR
from src.canny_edge_detector import extract_chip_edges_pipeline
from src.sector_masker import apply_sector_mask

os.makedirs(RESULTS_DIR, exist_ok=True)
output_img_path = str(RESULTS_DIR / "step3.5_sector_mask_comparison.png")

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
    
    # 1. Extract raw Canny edges
    _, edges = extract_chip_edges_pipeline(bgr)

    # 2. Apply Sector Mask around tool tip (~150, 150)
    masked_edges, mask = apply_sector_mask(edges, tip_x=150, tip_y=150, cutting_radius=135)

    # 3. Create Mask visualization overlay
    mask_vis = bgr.copy()
    mask_vis[mask == 0] = (mask_vis[mask == 0] * 0.25).astype(np.uint8)  # Dim excluded regions
    cv2.circle(mask_vis, (150, 150), 5, (255, 50, 0), -1)  # Tool tip

    # 4. Create Final Isolated Chip Contour Overlay (Neon Green on BGR)
    contours, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    final_overlay = bgr.copy()
    cv2.drawContours(final_overlay, contours, -1, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.circle(final_overlay, (150, 150), 5, (255, 50, 0), -1)

    col1 = annotate(edges, "1. Step 3.4: Raw Canny Edges", "Has workpiece glare & bed", color=(255, 200, 100))
    col2 = annotate(mask_vis, "2. Cutting Sector Mask", "Active shear zone (Lit up)", color=(100, 200, 255))
    col3 = annotate(masked_edges, "3. Step 3.5: Masked Edges", "Workpiece glare eradicated", color=(0, 255, 0))
    col4 = annotate(final_overlay, "4. Isolated Chip Overlay", "Green = Pure chip contour", color=(0, 255, 255))

    row = np.hstack([col1, col2, col3, col4])
    rows.append(row)

grid = np.vstack(rows)

header = np.zeros((50, grid.shape[1], 3), dtype=np.uint8) + 20
cv2.putText(header, "STEP 3.5: ASYMMETRIC SECTOR MASKING (Eliminates Workpiece Glare & Bed)", (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 220, 255), 2, cv2.LINE_AA)

final_image = np.vstack([header, grid])
cv2.imwrite(output_img_path, final_image)
print(f"Step 3.5 comparison successfully saved to: {output_img_path}")
