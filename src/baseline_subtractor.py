# -*- coding: utf-8 -*-
"""
Module: baseline_subtractor.py
Layer 4 (Phase 3 v2) — Adaptive Background Model

Replaces the fragile 3-frame static-intersection baseline with an adaptive Gaussian
Mixture Model (cv2.createBackgroundSubtractorMOG2) or K-Nearest Neighbours (KNN).

Why this is needed (planv2.md):
  The v1 static-intersection baseline only subtracted edges present in all 3 frames (t-3, t-2, t-1).
  Accumulated floor debris or stationary swarf on the tool shank that shifts slightly due to
  vibration was never identical across all 3 frames and leaked through.
  MOG2 continuously updates a per-pixel Gaussian mixture background model over time (history=90),
  naturally absorbing settled debris, tool shank, and machine bed into the background regardless
  of vibration jitter, while keeping actively moving chip foregrounds crisp.

Fusion rule:
  E_dynamic = E_masked AND foreground_mask

Public API
----------
AdaptiveBackgroundModel(method="MOG2", history=90, var_threshold=16, detect_shadows=False)
    .apply(roi_bgr, learning_rate=-1) -> foreground_mask
    .filter_edges(masked_edges, roi_bgr, learning_rate=-1) -> (dynamic_edges, fg_mask)
"""

import cv2
import numpy as np


class AdaptiveBackgroundModel:
    """
    Stateful per-video Adaptive Background Subtractor for Phase 3 Layer 4.
    Maintains a continuous background distribution model across consecutive frames of a video.
    """

    def __init__(
        self,
        method: str = "MOG2",
        history: int = 90,
        var_threshold: float = 16.0,
        detect_shadows: bool = False,
    ):
        """
        Parameters
        ----------
        method : str
            'MOG2' (Mixture of Gaussians, default) or 'KNN' (K-Nearest Neighbours).
        history : int
            Number of previous frames affecting the background model (~22s at 4 FPS). Default 90.
        var_threshold : float
            Threshold on Mahalanobis distance to decide whether pixel is foreground. Default 16.0.
        detect_shadows : bool
            Whether to detect and mark shadows. Default False (pure binary 0/255).
        """
        self.method = method.upper()
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows

        if self.method == "KNN":
            self.subtractor = cv2.createBackgroundSubtractorKNN(
                history=self.history,
                dist2Threshold=self.var_threshold * 25.0,
                detectShadows=self.detect_shadows,
            )
        else:
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.var_threshold,
                detectShadows=self.detect_shadows,
            )

        self.frames_seen = 0

    def reset(self):
        """Resets the internal background model state for a new video sequence."""
        if self.method == "KNN":
            self.subtractor = cv2.createBackgroundSubtractorKNN(
                history=self.history,
                dist2Threshold=self.var_threshold * 25.0,
                detectShadows=self.detect_shadows,
            )
        else:
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.var_threshold,
                detectShadows=self.detect_shadows,
            )
        self.frames_seen = 0

    def apply(self, roi_bgr: np.ndarray, learning_rate: float = -1.0) -> np.ndarray:
        """
        Updates the background model with the current ROI frame and returns the foreground mask.

        Parameters
        ----------
        roi_bgr : np.ndarray
            Input ROI frame (BGR uint8, 300x300).
        learning_rate : float
            Value between 0 and 1 indicating how fast the background model is updated.
            -1 (default) means an automatically chosen learning rate.

        Returns
        -------
        fg_mask : np.ndarray
            Binary uint8 mask (255 = dynamic foreground, 0 = background).
        """
        raw_mask = self.subtractor.apply(roi_bgr, learningRate=learning_rate)
        self.frames_seen += 1

        # Clean binary mask (eliminate shadow grayscale values if any)
        _, fg_mask = cv2.threshold(raw_mask, 127, 255, cv2.THRESH_BINARY)

        # Light 3x3 morphological closing to bridge micro-gaps inside solid chip bodies
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        return fg_mask

    def filter_edges(
        self,
        masked_edges: np.ndarray,
        roi_bgr: np.ndarray,
        learning_rate: float = -1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Executes Layer 4 edge filtering:
            E_dynamic = E_masked AND foreground_mask

        Parameters
        ----------
        masked_edges : np.ndarray
            Input edge map from Layer 3 (sector masked).
        roi_bgr : np.ndarray
            Current ROI crop in BGR format (used to update background model).
        learning_rate : float
            Learning rate for background subtractor update.

        Returns
        -------
        dynamic_edges : np.ndarray
            Filtered binary edge map with static machine bed/debris removed.
        fg_mask : np.ndarray
            The foreground mask generated by the adaptive model.
        """
        fg_mask = self.apply(roi_bgr, learning_rate=learning_rate)

        # Dilate foreground mask slightly (3x3) so edge pixels near the foreground boundary aren't clipped
        k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        fg_mask_dilated = cv2.dilate(fg_mask, k_dilate, iterations=1)

        dynamic_edges = cv2.bitwise_and(masked_edges, fg_mask_dilated)
        return dynamic_edges, fg_mask
