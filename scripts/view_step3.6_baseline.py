# -*- coding: utf-8 -*-
"""
Generate a side-by-side comparison image of Step 3.6 (Pre-Machining Baseline Frame 0 Subtraction)
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
from src.baseline_subtractor import compute_dilated_baseline, subtract_baseline_edges

os.makedirs(RESULTS_DIR, exist_ok=True)
output_img_path = str(RESULTS_DIR / "step3.6_baseline_subtraction_comparison.png")

samples = [
    ("Aluminium (Vid 01)",
     str(PROJECT_ROOT / "data/phase2_roi_crops/retained/WhatsApp_Video_2026-08-11_at_6.55.09_AM/crop_000000.jpg"),
     str(PROJECT_ROOT / "data/phase2_roi_crops/retained/WhatsApp_Video_2026-08-11_at_6.55.09_AM/crop_000450.jpg")),
    ("Copper (Vid 11)",
     str(PROJECT_ROOT / "data/phase2_roi_crops/retained/WhatsApp_Video_2026-08-11_at_6.55.14_AM_(1)/crop_000000.jpg"),
     str(PROJECT_ROOT / "data/phase2_roi_crops/retained/WhatsApp_Video_2026-08-11_at_6.55.14_AM_(1)/crop_000300.jpg")),
    ("Mild Steel (Vid 16)",
     str(PROJECT_ROOT / "data/phase2_roi_crops/retained/WhatsApp_Video_2026-08-11_at_6.55.16_AM_(1)/crop_000000.jpg"),
     str(PROJECT_ROOT / "data/phase2_roi_crops/retained/WhatsApp_Video_2026-08-11_at_6.55.16_AM_(1)/crop_000450.jpg"))
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
for material_label, f0_path, ft_path in samples:
    f0_bgr = cv2.imread(f0_path)
    ft_bgr = cv2.imread(ft_path)
    if f0_bgr is None or ft_bgr is None:
        continue
    
    # 1. Compute dilated Frame 0 baseline
    _, e0_dilated = compute_dilated_baseline(f0_bgr, tip_x=150, tip_y=150, kernel_size=3, iterations=1)

    # 2. Extract current frame masked edges
    _, et_raw = extract_chip_edges_pipeline(ft_bgr)
    et_masked, _ = apply_sector_mask(et_raw, tip_x=150, tip_y=150, cutting_radius=135)

    # 3. Step 3.6: Subtract dilated static baseline E0
    dynamic_chip_edges = subtract_baseline_edges(et_masked, e0_dilated)

    # 4. Highlight dynamic chip overlay (Neon Green)
    contours, _ = cv2.findContours(dynamic_chip_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    overlay = ft_bgr.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.circle(overlay, (150, 150), 5, (255, 50, 0), -1)

    col1 = annotate(f0_bgr, "1. Frame 0 (Static Baseline)", "Pre-cutting machine reference", color=(255, 200, 100))
    col2 = annotate(ft_bgr, "2. Active Machining Frame", "Tool cutting metal chips", color=(255, 255, 255))
    col3 = annotate(et_masked, "3. Step 3.5: Masked Edges", "Contains static tool shank & clamps", color=(100, 200, 255))
    col4 = annotate(overlay, "4. Step 3.6: Dynamic Chip Overlay", "Static bed erased | Pure chip", color=(0, 255, 0))

    row = np.hstack([col1, col2, col3, col4])
    rows.append(row)

grid = np.vstack(rows)

header = np.zeros((50, grid.shape[1], 3), dtype=np.uint8) + 20
cv2.putText(header, "STEP 3.6: PRE-MACHINING FRAME 0 BASELINE SUBTRACTION (Erases Static Bed)", (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 220, 255), 2, cv2.LINE_AA)

final_image = np.vstack([header, grid])
cv2.imwrite(output_img_path, final_image)
print(f"Step 3.6 comparison successfully saved to: {output_img_path}")
