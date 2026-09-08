# -*- coding: utf-8 -*-
"""
Demonstration: Isolating ONLY the chip at the tool tip and erasing workpiece glare/bed
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import PROJECT_ROOT, RESULTS_DIR
from src.lab_preprocessor import get_lightness_channel, apply_clahe

os.makedirs(RESULTS_DIR, exist_ok=True)
output_path = str(RESULTS_DIR / "step_isolated_chip_demo.png")

# Load sample with copper chips at tool tip
sample_path = str(PROJECT_ROOT / "data/phase2_roi_crops/retained/WhatsApp_Video_2026-08-11_at_6.55.14_AM_(1)/crop_000150.jpg")
img_bgr = cv2.imread(sample_path)
H, W = img_bgr.shape[:2]

# Tool tip is at center (~150, 150) in 300x300 crop
tip_x, tip_y = 150, 150

# 1. CLAHE Lightness
l_raw = get_lightness_channel(img_bgr)
l_clahe = apply_clahe(l_raw, clip_limit=2.5, tile_grid_size=(8, 8))

# 2. Canny Edge Detection
v = np.median(l_clahe)
edges = cv2.Canny(l_clahe, int(max(0, 0.66 * v)), int(min(255, 1.33 * v)))

# 3. Spatial Sector & Proximity Mask around Tool Tip (Cutting Zone)
# Keep only region within cutting radius (e.g. 140px) and exclude upper workpiece cylinder glare
proximity_mask = np.zeros((H, W), dtype=np.uint8)
cv2.circle(proximity_mask, (tip_x, tip_y), 130, 255, -1)

# Mask out upper-left cylinder region where workpiece glare lives
glare_suppression_mask = np.ones((H, W), dtype=np.uint8) * 255
# Workpiece cylinder above cutting edge
cv2.rectangle(glare_suppression_mask, (0, 0), (tip_x - 10, tip_y - 20), 0, -1)

# Combined cutting zone mask
cutting_zone = cv2.bitwise_and(proximity_mask, glare_suppression_mask)
chip_edges_filtered = cv2.bitwise_and(edges, edges, mask=cutting_zone)

# 4. Extract Contours & Overlay on Color Image
contours, _ = cv2.findContours(chip_edges_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

overlay = img_bgr.copy()
# Draw isolated active chips in neon green
cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2, lineType=cv2.LINE_AA)
# Draw Blue Dot on tool tip
cv2.circle(overlay, (tip_x, tip_y), 5, (255, 50, 0), -1)

def annotate(img, text, subtext="", color=(255, 255, 255)):
    canvas = img.copy()
    if len(canvas.shape) == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 45), (20, 20, 20), -1)
    cv2.putText(canvas, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    if subtext:
        cv2.putText(canvas, subtext, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
    return canvas

p1 = annotate(img_bgr, "1. Raw 300x300 Crop", "Shows workpiece glare & tool")
p2 = annotate(edges, "2. All Detected Edges", "Includes fake workpiece glare lines", color=(0, 0, 255))
p3 = annotate(chip_edges_filtered, "3. Workpiece Glare Erased", "Only active chips at tool tip", color=(0, 255, 255))
p4 = annotate(overlay, "4. Segmented Chip (Neon Green)", "Pure chip isolated for measurement", color=(0, 255, 0))

panel = np.hstack([p1, p2, p3, p4])
cv2.imwrite(output_path, panel)
print("Saved suppression demo to:", output_path)
