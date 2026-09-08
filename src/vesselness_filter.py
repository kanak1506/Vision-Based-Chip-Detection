# -*- coding: utf-8 -*-
"""
Module: vesselness_filter.py
Layer 2b & 2c (Phase 3 v2) -- Frangi Vesselness Ridge Channel & Frangi-Guided Canny Fusion

Implements:
1. Multi-scale Frangi curvilinear ridge filter (Frangi et al. 1998) on the L-channel (Layer 2b)
   to detect continuous tubular/curvilinear chip geometries via Hessian eigenvalues.
2. Frangi-Guided Canny Selective Fusion (Layer 2c):
   Uses the detected Frangi curvilinear structures as a spatial attention mask.
   Expands the Frangi binary mask by a small structuring element (vicinity tolerance) and
   retains ONLY Canny edges that lie within this spatial neighborhood:
       Final edges = Canny edges AND Dilated_Frangi_Vicinity_Mask

This eliminates background lathe bed clutter, machine textures, and chip tray gravel
while preserving fine Canny edge localization along active chip ribbons.

Public API
----------
apply_vesselness_filter(l_denoised, sigmas, alpha, beta, gamma, percentile_thresh)
    -> (vesselness_map, ridge_binary)

fuse_canny_vesselness(canny_edges, ridge_binary, vicinity_ksize)
    -> E_combined  (uint8 binary, 255 = edge within Frangi vicinity, 0 = background)

compute_vesselness_and_fuse(l_denoised, canny_edges, ...)
    -> (vesselness_map, ridge_binary, E_combined)
"""

import cv2
import numpy as np
from skimage.filters import frangi


# ---------------------------------------------------------------------------
# Default parameters (planv2.md starting values, tune per material in Phase 5)
# ---------------------------------------------------------------------------
DEFAULT_SIGMAS          = range(1, 6)   # 1-5 px: covers ~2-6px chip strand width at 300x300 ROI
DEFAULT_ALPHA           = 0.5           # blob-ness suppression (skimage alpha, plan beta1 ~= 0.5)
DEFAULT_BETA            = 0.5           # shape anisotropy weight (skimage beta)
DEFAULT_GAMMA           = None          # auto-scale to image's Hessian norm
DEFAULT_PERCENTILE      = 92.0          # top 8% of vesselness response (lowered to retain fainter curvilinear structures)
DEFAULT_VICINITY_KSIZE  = 7             # 7x7 elliptical structuring element (~3px spatial dilation radius)


def apply_vesselness_filter(
    l_denoised: np.ndarray,
    sigmas=DEFAULT_SIGMAS,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    percentile_thresh: float = DEFAULT_PERCENTILE,
) -> tuple:
    """
    Runs multi-scale Frangi filter on the denoised L-channel and thresholds
    the response map to produce a binary ridge map.

    Parameters
    ----------
    l_denoised : np.ndarray
        Denoised lightness channel (uint8, 0-255) from Layer 1
        (extract_chip_edges_pipeline's first return value).
    sigmas : iterable of int/float
        Scale range in pixels. Default range(1, 6) covers 1-5 px.
    alpha : float
        Blob-ness suppression weight. Default 0.5.
    beta : float
        Anisotropy weight. Default 0.5.
    gamma : float
        Structureness sensitivity. None = auto (half of max Hessian norm).
    percentile_thresh : float
        Percentile used for thresholding. Pixels above this percentile of the
        vesselness response are kept as ridges (default: 92.0 -> top 8%).

    Returns
    -------
    vesselness_map : np.ndarray
        Float64 vesselness response in [0, 1], same H x W as input.
    ridge_binary : np.ndarray
        Binary uint8 map (255 = ridge, 0 = background), same H x W as input.
    """
    # Normalize to float [0, 1] -- frangi expects float input
    l_float = l_denoised.astype(np.float64) / 255.0

    # Multi-scale Frangi filter.
    # black_ridges=False: chip strands are BRIGHT ridges on darker background.
    vesselness_map = frangi(
        l_float,
        sigmas=sigmas,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=False,
        mode="reflect",
    )

    # Threshold: keep top (100 - percentile_thresh)% of vesselness response.
    thresh = float(np.percentile(vesselness_map, percentile_thresh))
    if thresh <= 0.0:
        # Otsu fallback on uint8-scaled map
        v_uint8 = (vesselness_map * 255.0).clip(0, 255).astype(np.uint8)
        otsu_thresh, _ = cv2.threshold(v_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = float(otsu_thresh) / 255.0

    ridge_binary = np.where(vesselness_map >= thresh, np.uint8(255), np.uint8(0)).astype(np.uint8)

    return vesselness_map, ridge_binary


def fuse_canny_vesselness(
    canny_edges: np.ndarray = None,
    ridge_binary: np.ndarray = None,
    vicinity_ksize: int = DEFAULT_VICINITY_KSIZE,
) -> np.ndarray:
    """
    Pure Frangi Vesselness Edge Representation (Canny completely bypassed):
    Returns the binary Frangi curvilinear ridge map directly for downstream processing.
    """
    if ridge_binary is None:
        if canny_edges is not None:
            return np.zeros_like(canny_edges)
        return np.zeros((300, 300), dtype=np.uint8)

    return ridge_binary.copy()


def compute_vesselness_and_fuse(
    l_denoised: np.ndarray,
    canny_edges: np.ndarray = None,
    sigmas=DEFAULT_SIGMAS,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    percentile_thresh: float = DEFAULT_PERCENTILE,
    vicinity_ksize: int = DEFAULT_VICINITY_KSIZE,
) -> tuple:
    """
    Convenience wrapper: runs apply_vesselness_filter and outputs pure Frangi vesselness edges.
    This is the single call used by the pipeline runner.

    Parameters
    ----------
    l_denoised : np.ndarray
        Denoised L-channel from Layer 1.
    canny_edges : np.ndarray, optional
        Unused, kept for backwards compatibility.
    (remaining params identical to apply_vesselness_filter)

    Returns
    -------
    vesselness_map : np.ndarray  -- float64 [0, 1] response
    ridge_binary   : np.ndarray  -- uint8 binary ridge map (Layer 2b)
    E_combined     : np.ndarray  -- uint8 pure Frangi vesselness edge map (Layer 2c)
    """
    vesselness_map, ridge_binary = apply_vesselness_filter(
        l_denoised, sigmas, alpha, beta, gamma, percentile_thresh
    )
    E_combined = ridge_binary.copy()
    return vesselness_map, ridge_binary, E_combined
