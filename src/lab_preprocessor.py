# -*- coding: utf-8 -*-
"""
Module: lab_preprocessor.py
Step 1: Decouples Luminance (Lightness) from Chrominance (Color) using CIELAB space.
Step 2: Smart Local Lighting Equalization via CLAHE on the Lightness (L) channel.
Step 3: Edge-Preserving Denoising via Bilateral Filtering on the equalized Lightness (L) channel.
"""

import cv2
import numpy as np


def bgr_to_lab(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts BGR image to CIELAB and returns separated (L, A, B) 2D channels."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    return cv2.split(lab)


def lab_to_bgr(l: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Merges (L, A, B) 2D channels back into a BGR image."""
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def get_lightness_channel(image_bgr: np.ndarray) -> np.ndarray:
    """Extracts pure Lightness (L) 2D array [0-255]."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]


def apply_clahe(l_channel: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Applies Contrast-Limited Adaptive Histogram Equalization on the Lightness (L) channel."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(l_channel)


def apply_bilateral_filter(l_channel: np.ndarray, d: int = 5, sigma_color: float = 50.0, sigma_space: float = 50.0) -> np.ndarray:
    """Smooths sensor noise and surface grain while strictly preserving sharp chip boundaries."""
    return cv2.bilateralFilter(l_channel, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def preprocess_lightness_pipeline(image_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Executes Step 1 + Step 2 + Step 3: Returns fully normalized & denoised Lightness (L) channel."""
    l_raw = get_lightness_channel(image_bgr)
    l_clahe = apply_clahe(l_raw, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    return apply_bilateral_filter(l_clahe)


def normalize_illumination_bgr(image_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Applies CLAHE illumination normalization to the L-channel while preserving original color channels (A, B)."""
    l, a, b = bgr_to_lab(image_bgr)
    l_clahe = apply_clahe(l, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    return lab_to_bgr(l_clahe, a, b)

