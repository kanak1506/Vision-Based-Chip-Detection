# -*- coding: utf-8 -*-
"""
Generate a side-by-side comparison image of Step 3.8 (Cutting Zone Proximity & Clean Contour Extraction)
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
from src.specular_filter import filter_specular_lines
from src.contour_extractor import extract_chip_contours

os.makedirs(RESULTS_DIR, exist_ok=True)
output_img_path = str(RESULTS_DIR / "step3.8_contour_extraction_comparison.png")

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
    
    # 1. Baseline
    _, e0_dilated = compute_dilated_baseline(f0_bgr, tip_x=150, tip_y=150)
    # 2. Canny + Sector Mask
    _, et_raw = extract_chip_edges_pipeline(ft_bgr)
    et_masked, _ = apply_sector_mask(et_raw, tip_x=150, tip_y=150, cutting_radius=135)
    # 3. Pure Uniform Whole-Frame Baseline Subtraction
    e_subtracted = subtract_baseline_edges(et_masked, e0_dilated)
    # 4. Specular Line Filter
    chip_edges_clean, _ = filter_specular_lines(e_subtracted, tip_x=150, tip_y=150, max_linearity_residual=1.6, min_line_len=10)
    # 5. Step 3.8 Tool-Tip Seeded Connected Ribbon Tracking
    chip_mask, contours, metrics = extract_chip_contours(chip_edges_clean, tip_x=150, tip_y=150, max_dist_to_tip=135.0, min_contour_length=14, max_link_dist=8.0)

    # 6. Clean 1px Overlay on BGR
    overlay = ft_bgr.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    # Proximity circle
    cv2.circle(overlay, (150, 150), 135, (100, 200, 255), 1, lineType=cv2.LINE_AA)
    # Tool tip dot
    cv2.circle(overlay, (150, 150), 5, (255, 50, 0), -1)

    col1 = annotate(ft_bgr, f"{material_label}: Raw Crop", "Active machining frame", color=(255, 255, 255))
    col2 = annotate(chip_edges_clean, "Step 3.7: Clean Edges", "No glare / No bed lines", color=(255, 200, 100))
    col3 = annotate(chip_mask, "Step 3.8: Binary Chip Mask", f"{metrics['num_chip_clusters']} clusters | Area: {int(metrics['total_chip_area'])}px", color=(100, 200, 255))
    col4 = annotate(overlay, "Step 3.8: Segmented Overlay", "Neon Green = Tracked Chip Contours", color=(0, 255, 0))

    row = np.hstack([col1, col2, col3, col4])
    rows.append(row)

grid = np.vstack(rows)

header = np.zeros((50, grid.shape[1], 3), dtype=np.uint8) + 20
cv2.putText(header, "STEP 3.8: CUTTING ZONE PROXIMITY & CLEAN CONTOUR EXTRACTION", (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 220, 255), 2, cv2.LINE_AA)

final_image = np.vstack([header, grid])
cv2.imwrite(output_img_path, final_image)
print(f"Step 3.8 comparison successfully saved to: {output_img_path}")
