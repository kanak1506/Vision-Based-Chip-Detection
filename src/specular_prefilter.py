# -*- coding: utf-8 -*-
"""
Module: specular_prefilter.py
Layer 0 (Phase 3 v2) — Dichromatic Specular Pre-Filter

Removes diffuse glare patches from the raw ROI crop *before* any edge or
ridge detector sees the image.  Based on the dichromatic reflection model
(Tan & Ikeuchi 2005; Shen & Cai 2009): specular highlights have low
saturation and high intensity, which separates them reliably from both the
workpiece body and the chip itself.

Flagged pixels are inpainted (Telea method) so downstream detectors never
see the glare as a real boundary — complementary to Layer 5's geometric
line-rejection, which only catches specular highlights that survive as thin
straight edges.

Public API
----------
apply_specular_prefilter(roi_bgr, s_thresh, v_thresh, inpaint_radius)
    -> (inpainted_bgr, specular_mask, specular_pct)

The returned specular_pct should be logged to the per-frame manifest.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Default threshold values (0-255 scale).
# Calibrate per material on a validation batch; these are the starting point
# recommended in planv2.md.
# ---------------------------------------------------------------------------
DEFAULT_S_THRESH: int = 40    # pixels with S < this are specular candidates
DEFAULT_V_THRESH: int = 220   # pixels with V > this are specular candidates
DEFAULT_INPAINT_RADIUS: int = 5  # Telea inpaint neighbourhood radius (px)


def build_specular_mask(
    roi_bgr: np.ndarray,
    s_thresh: int = DEFAULT_S_THRESH,
    v_thresh: int = DEFAULT_V_THRESH,
) -> tuple:
    """
    Identifies specular-highlight pixels using the dichromatic reflection model.

    A pixel is flagged as specular if:
        S < s_thresh  (low saturation - specular lobe washes out hue)
        AND
        V > v_thresh  (high brightness - specular lobe is bright)

    Parameters
    ----------
    roi_bgr : np.ndarray
        Input ROI crop in BGR format, uint8.
    s_thresh : int
        Saturation upper bound (0-255).  Default 40.
    v_thresh : int
        Value (brightness) lower bound (0-255).  Default 220.

    Returns
    -------
    mask : np.ndarray
        Binary uint8 mask (255 = specular, 0 = normal), same H x W as input.
    specular_pct : float
        Percentage of pixels flagged (0.0-100.0).  Log this per frame.
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    mask = np.where(
        (s_channel.astype(np.int32) < s_thresh) & (v_channel.astype(np.int32) > v_thresh),
        np.uint8(255),
        np.uint8(0),
    ).astype(np.uint8)

    total_pixels = mask.size
    specular_pct = float(np.count_nonzero(mask)) / total_pixels * 100.0

    return mask, specular_pct


def inpaint_specular(
    roi_bgr: np.ndarray,
    mask: np.ndarray,
    inpaint_radius: int = DEFAULT_INPAINT_RADIUS,
) -> np.ndarray:
    """
    Inpaints masked pixels using the Telea algorithm.

    Telea's method propagates local neighbourhood texture smoothly into the
    masked region, which is preferred here because specular highlights often
    sit on top of curved chip or workpiece surfaces.

    Parameters
    ----------
    roi_bgr : np.ndarray
        Original ROI crop in BGR format.
    mask : np.ndarray
        Binary mask from build_specular_mask() (255 = pixels to repair).
    inpaint_radius : int
        Neighbourhood radius for Telea inpainting.  Default 5.

    Returns
    -------
    inpainted : np.ndarray
        BGR image with specular pixels replaced by inpainted content.
    """
    if not np.any(mask):
        # Nothing to inpaint - return original to avoid wasted work.
        return roi_bgr.copy()

    return cv2.inpaint(roi_bgr, mask, inpaint_radius, cv2.INPAINT_TELEA)


def apply_specular_prefilter(
    roi_bgr: np.ndarray,
    s_thresh: int = DEFAULT_S_THRESH,
    v_thresh: int = DEFAULT_V_THRESH,
    inpaint_radius: int = DEFAULT_INPAINT_RADIUS,
) -> tuple:
    """
    Full Layer-0 pipeline: detect specular pixels and inpaint them.

    This is the single entry-point called by the main pipeline runner
    *before* Layer 1 (lab_preprocessor).  The inpainted image replaces
    the raw crop for all subsequent layers.

    Parameters
    ----------
    roi_bgr : np.ndarray
        Raw ROI crop from Phase 2 (BGR, uint8).
    s_thresh : int
        Saturation threshold (dichromatic model lower bound).
    v_thresh : int
        Value threshold (dichromatic model upper bound).
    inpaint_radius : int
        Telea inpaint neighbourhood radius.

    Returns
    -------
    inpainted_bgr : np.ndarray
        Specular-cleaned ROI crop - feed this into Layer 1.
    specular_mask : np.ndarray
        Binary mask of flagged pixels - useful for QA visualisation.
    specular_pct : float
        Fraction of image that was specular (0-100).  Log to manifest.
        A sudden jump across frames signals a lighting change.
    """
    specular_mask, specular_pct = build_specular_mask(roi_bgr, s_thresh, v_thresh)
    inpainted_bgr = inpaint_specular(roi_bgr, specular_mask, inpaint_radius)
    return inpainted_bgr, specular_mask, specular_pct
