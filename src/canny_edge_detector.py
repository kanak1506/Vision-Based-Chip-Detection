# -*- coding: utf-8 -*-
"""
Module: canny_edge_detector.py
Step 4: Adaptive Canny Edge Detection with Automatic Median-Based Hysteresis Thresholding.
"""

import cv2
import numpy as np


def compute_adaptive_canny(image_gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """
    Computes adaptive Canny edge detection by dynamically deriving lower and upper
    hysteresis thresholds from the median intensity of the image.
    """
    v = float(np.median(image_gray))
    lower_thresh = int(max(0, (1.0 - sigma) * v))
    upper_thresh = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image_gray, lower_thresh, upper_thresh)


def extract_chip_edges_pipeline(image_bgr: np.ndarray, clip_limit: float = 2.5, sigma: float = 0.33) -> tuple[np.ndarray, np.ndarray]:
    """
    End-to-end Steps 1-4 pipeline:
    Raw BGR -> LAB (L) -> CLAHE -> Bilateral Denoising -> Adaptive Canny.
    Returns (l_denoised, edge_map).
    """
    # 1. Decouple Lightness (L)
    l_raw = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    # 2. CLAHE Equalization
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_raw)
    # 3. Bilateral Denoising
    l_denoised = cv2.bilateralFilter(l_clahe, d=5, sigmaColor=50, sigmaSpace=50)
    # 4. Adaptive Canny
    edges = compute_adaptive_canny(l_denoised, sigma=sigma)
    return l_denoised, edges
