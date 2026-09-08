# -*- coding: utf-8 -*-
"""
Module: overlay_renderer.py
Layer 8 (Phase 3 v2): Crisp Real-Time Chip Contour & Telemetry HUD Overlay.

Renders crisp 1px neon green chip contour outlines (with zero opaque/translucent tint wash),
tool-tip tracking crosshair, and real-time machining telemetry HUD banner.
"""

import cv2
import numpy as np


def render_chip_overlay(
    crop_bgr: np.ndarray,
    chip_mask: np.ndarray,
    tip_x: int = 150,
    tip_y: int = 150,
    proximity_radius: int = 140,
    alpha: float = 0.0,
    metrics: dict = None
) -> np.ndarray:
    """
    Renders crisp, clean neon green chip contour boundaries and telemetry HUD onto raw crop.
    Zero green tint wash: raw workpiece and tool remain 100% visible and unclouded.
    """
    canvas = crop_bgr.copy()
    h, w = canvas.shape[:2]

    # 1. Crisp Single-Line Centerline Spine (No boundary loop / border effect, Zero green wash)
    contours, _ = cv2.findContours(chip_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if alpha > 0:
        overlay = canvas.copy()
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), cv2.FILLED)
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)

    # Render 1.5px clean neon green single spine directly on canvas
    k_line = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated_spine = cv2.dilate(chip_mask, k_line, iterations=1)
    canvas[dilated_spine > 0] = [0, 255, 0]

    # 2. Tool-tip Proximity Envelope (Disabled per user preference)
    # if proximity_radius > 0:
    #     cv2.circle(canvas, (int(tip_x), int(tip_y)), proximity_radius, (0, 220, 255), 1, lineType=cv2.LINE_AA)

    # 3. Precision Tool-Tip Tracking Point (Blue dot removed per user preference)

    # 4. HUD Telemetry Banner
    if metrics is not None:
        cv2.rectangle(canvas, (0, h - 22), (w, h), (15, 15, 15), -1)
        clusters = metrics.get("num_chip_clusters", len(contours))
        area = int(metrics.get("total_chip_area", 0))
        hud_text = f"Chips: {clusters} | Area: {area}px"
        cv2.putText(canvas, hud_text, (6, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

    return canvas
