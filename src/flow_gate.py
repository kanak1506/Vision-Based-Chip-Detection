# -*- coding: utf-8 -*-
"""
Module: flow_gate.py
Layer 4b (Phase 3 v2) — Optical Flow Motion-Coherence Gating

Computes dense Farneback optical flow on consecutive grayscale ROI frames to extract
pixel-level motion magnitude and directional coherence. Gates dynamic edges from Layer 4
to discard stationary debris, residual workpiece glare lines, and non-moving vibration artifacts.

Why this is needed (planv2.md):
  Static floor debris, settled chips, and tool clamping shadows have zero true motion
  (motion magnitude M ~ 0 px/frame). In contrast, living chips being sheared off at the
  cutting zone accelerate and eject at high velocities (M >> 1.0 px/frame).
  Layer 4b computes dense Farneback optical flow between consecutive frames:
    M = sqrt(u^2 + v^2)
    motion_mask = (M >= min_magnitude)
  Edges are gated as:
    E_gated = E_dynamic AND motion_mask

  To protect the immediate cutting shear initiation zone at the tool tip (where chip velocity
  starts from zero before curling), a circular proximity protection zone is preserved.

Public API
----------
OpticalFlowGate(min_magnitude=0.8, tip_protect_radius=30, mask_dilation_ksize=5)
    .apply(curr_gray, dynamic_edges, tip_x=150, tip_y=150) -> (gated_edges, motion_mask, flow)
    .reset()
"""

import cv2
import numpy as np


class OpticalFlowGate:
    """
    Stateful per-video Optical Flow Motion-Coherence Gater for Phase 3 Layer 4b.
    Tracks consecutive frames to compute velocity fields and filter stationary edges.
    """

    def __init__(
        self,
        min_magnitude: float = 1.2,
        tip_protect_radius: int = 25,
        mask_dilation_ksize: int = 5,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
    ):
        """
        Parameters
        ----------
        min_magnitude : float
            Minimum motion magnitude in pixels/frame required to pass the gate. Default 1.2.
        tip_protect_radius : int
            Radius around tool tip where edges are always preserved (shear initiation protection). Default 25.
        mask_dilation_ksize : int
            Structuring element size to dilate motion mask so thin edge spines are not clipped. Default 5.
        """
        self.min_magnitude = min_magnitude
        self.tip_protect_radius = tip_protect_radius
        self.mask_dilation_ksize = mask_dilation_ksize

        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma

        self.prev_gray: np.ndarray | None = None
        self.frame_idx = 0

    def reset(self):
        """Resets the previous frame state for a new video sequence."""
        self.prev_gray = None
        self.frame_idx = 0

    def compute_flow(self, curr_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes dense Farneback optical flow and returns (flow, magnitude, angle).
        """
        if self.prev_gray is None or self.prev_gray.shape != curr_gray.shape:
            # First frame passthrough
            flow = np.zeros((curr_gray.shape[0], curr_gray.shape[1], 2), dtype=np.float32)
            magnitude = np.full(curr_gray.shape[:2], 999.0, dtype=np.float32)  # Passthrough
            angle = np.zeros(curr_gray.shape[:2], dtype=np.float32)
            return flow, magnitude, angle

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            curr_gray,
            None,
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            flags=0,
        )

        u = flow[..., 0]
        v = flow[..., 1]
        magnitude = np.sqrt(u**2 + v**2)
        angle = np.arctan2(v, u) * (180.0 / np.pi)

        return flow, magnitude, angle

    def apply(
        self,
        curr_gray_or_bgr: np.ndarray,
        dynamic_edges: np.ndarray,
        tip_x: int = 150,
        tip_y: int = 150,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes Layer 4b Optical Flow Motion-Coherence Gating:
            E_gated = E_dynamic AND motion_mask (with tool-tip shear zone protected)

        Parameters
        ----------
        curr_gray_or_bgr : np.ndarray
            Current ROI crop (grayscale or BGR uint8, 300x300).
        dynamic_edges : np.ndarray
            Input edge map from Layer 4 (adaptive background subtracted).
        tip_x : int
            Tool tip x coordinate in ROI crop.
        tip_y : int
            Tool tip y coordinate in ROI crop.

        Returns
        -------
        gated_edges : np.ndarray
            Motion-gated binary edge map.
        motion_mask : np.ndarray
            Binary motion mask (255 = moving/protected, 0 = stationary).
        magnitude : np.ndarray
            Float32 optical flow velocity magnitude field in px/frame.
        """
        if len(curr_gray_or_bgr.shape) == 3:
            curr_gray = cv2.cvtColor(curr_gray_or_bgr, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = curr_gray_or_bgr.copy()

        flow, magnitude, angle = self.compute_flow(curr_gray)

        if self.prev_gray is None:
            # First frame: pass all edges, store previous gray
            self.prev_gray = curr_gray
            self.frame_idx += 1
            motion_mask = np.full(dynamic_edges.shape[:2], 255, dtype=np.uint8)
            return dynamic_edges.copy(), motion_mask, magnitude

        # 1. Motion Magnitude Mask: Pixels moving faster than threshold (M >= min_magnitude)
        motion_mask = np.where(magnitude >= self.min_magnitude, np.uint8(255), np.uint8(0))

        # 2. Shear Root Protection: Protect immediate shear initiation zone (r <= 25px from tool tip)
        # All outer regions (including tool post/carriage) must pass optical flow motion checking (M >= min_magnitude)
        cv2.circle(motion_mask, (int(tip_x), int(tip_y)), 25, 255, -1)

        # 3. Morphological Dilation on Motion Mask to ensure thin edge boundaries are covered
        if self.mask_dilation_ksize > 1:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.mask_dilation_ksize, self.mask_dilation_ksize))
            motion_mask = cv2.dilate(motion_mask, k, iterations=1)

        # 4. Gating: E_gated = E_dynamic AND motion_mask
        gated_edges = cv2.bitwise_and(dynamic_edges, motion_mask)

        # Update state
        self.prev_gray = curr_gray
        self.frame_idx += 1

        return gated_edges, motion_mask, magnitude
