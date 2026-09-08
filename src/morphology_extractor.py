# -*- coding: utf-8 -*-
"""
Module: morphology_extractor.py
Step 3.10: Contour Extraction & Morphology Feature Metrics for ISO 3685 Chip Shape Classification.
Computes Circularity, Aspect Ratio, Solidity, Skeleton Length, and ISO 3685 Class.
"""

import cv2
import numpy as np


def compute_morphology_metrics(contours: list, shape: tuple = (300, 300)) -> dict:
    """
    Computes comprehensive morphological features across all active chip contours:
      - circularity: 4*pi*Area / Perimeter^2 (High for C-chips, low for ribbons)
      - mean_aspect_ratio: Ratio of major to minor bounding axis
      - mean_solidity: Contour area / Convex hull area
      - skeleton_length: Total thinned strand length in pixels
      - iso_3685_class: Predicted ISO 3685 Chip Shape Class
    """
    if not contours:
        return {
            "total_area": 0.0,
            "total_perimeter": 0.0,
            "mean_circularity": 0.0,
            "mean_aspect_ratio": 1.0,
            "mean_solidity": 0.0,
            "skeleton_length": 0,
            "num_clusters": 0,
            "iso_3685_class": "NO_CHIP"
        }

    areas = []
    perimeters = []
    circularities = []
    aspect_ratios = []
    solidities = []

    # Rasterize contours for skeletonization
    mask = np.zeros(shape[:2], dtype=np.uint8)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        areas.append(area)
        perimeters.append(perimeter)

        # Circularity (Compactness)
        c = (4.0 * np.pi * area) / (perimeter ** 2)
        circularities.append(min(1.0, c))

        # Convex Hull & Solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(1.0, hull_area)
        solidities.append(min(1.0, solidity))

        # Rotated Bounding Box & Aspect Ratio
        if len(cnt) >= 5:
            rect = cv2.minAreaRect(cnt)
            (w_box, h_box) = rect[1]
            ar = max(w_box, h_box) / max(1.0, min(w_box, h_box))
            aspect_ratios.append(ar)
        else:
            aspect_ratios.append(1.0)

        cv2.drawContours(mask, [cnt], -1, 255, thickness=1)

    # Skeleton strand length
    skeleton_len = int(np.count_nonzero(mask))

    total_a = sum(areas)
    total_p = sum(perimeters)
    mean_c = float(np.mean(circularities)) if circularities else 0.0
    mean_ar = float(np.mean(aspect_ratios)) if aspect_ratios else 1.0
    mean_sol = float(np.mean(solidities)) if solidities else 0.0

    # Rule-Based ISO 3685 Chip Shape Classification
    if total_a < 30 and len(contours) <= 2:
        iso_class = "ELEMENTAL_DISCONTINUOUS"
    elif mean_c >= 0.35 or mean_sol >= 0.65:
        iso_class = "C_SHAPED_SEGMENTED"
    elif mean_ar >= 4.0 or mean_c < 0.12:
        iso_class = "RIBBON_CONTINUOUS"
    elif 0.12 <= mean_c < 0.25:
        iso_class = "TUBULAR_HELICAL"
    else:
        iso_class = "SPIRAL_COIL"

    return {
        "total_area": round(total_a, 2),
        "total_perimeter": round(total_p, 2),
        "mean_circularity": round(mean_c, 3),
        "mean_aspect_ratio": round(mean_ar, 2),
        "mean_solidity": round(mean_sol, 3),
        "skeleton_length": skeleton_len,
        "num_clusters": len(contours),
        "iso_3685_class": iso_class
    }
