# -*- coding: utf-8 -*-
"""
Module: specular_filter.py
Layer 5 (Phase 3 v2) — Workpiece Specular Reflection Line & Corner Filtering

Erases:
  1. Horizontal workpiece specular reflection streaks (orthogonal residual <= 1.6 px, R^2 > 0.95).
  2. Vertical workpiece shoulder / tool flank edges.
  3. Rigid angle-connected tool insert corners (approxPolyDP vertex angle filtering: 45°-135°).
  4. Cylindrical reflection bands in the upper workpiece zone (Fix 2.2).

Fixes applied per planv2.md:
  - [Fix 2.1] Expanded tool-tip shear-zone protected box:
      x in [tip_x - 25, tip_x + 35], y in [tip_y - 25, tip_y + 20] (60x45 px asymmetric)
      Covers the entire shear initiation root and curling zone so live chips are never clipped.
  - [Fix 2.2] Cylindrical reflection band suppression:
      Detects horizontal reflection streaks in y in [0, tip_y - 15] with R^2 > 0.92, |theta| < 15°,
      and distance to tip > 25 px.

Public API
----------
filter_specular_lines(edge_map, tip_x=150, tip_y=150, max_linearity_residual=1.6, min_line_len=12)
    -> (filtered_chip_edges, removed_specular_map)
"""

import cv2
import numpy as np


def is_rigid_tool_corner(approx_pts: np.ndarray, min_segment_len: float = 10.0) -> bool:
    """
    Checks if an approximated contour polygon matches a rigid machine tool corner
    (2 or 3 straight line segments meeting at sharp angles: ~60 deg, ~80 deg, ~90 deg).
    """
    if len(approx_pts) < 3 or len(approx_pts) > 6:
        return False

    pts = approx_pts.reshape(-1, 2)
    for i in range(len(pts) - 2):
        p0, p1, p2 = pts[i].astype(np.float32), pts[i + 1].astype(np.float32), pts[i + 2].astype(np.float32)
        v1 = p1 - p0
        v2 = p2 - p1

        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)

        if len1 < min_segment_len or len2 < min_segment_len:
            continue

        # Compute corner angle
        cos_theta = np.dot(v1, v2) / (len1 * len2 + 1e-6)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_theta))

        # Sharp machine tool corner: 45° to 135° turning angle
        if 45.0 <= angle_deg <= 135.0:
            return True

    return False


def compute_linearity_r_squared(pts: np.ndarray) -> tuple[float, float, float]:
    """
    Computes R^2 linearity score and orientation angle of a 2D point set.
    Returns (r_squared, angle_deg, orthogonal_residual).
    """
    if len(pts) < 5:
        return 0.0, 0.0, 999.0

    [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy = float(vx[0]), float(vy[0])
    x0, y0 = float(x0[0]), float(y0[0])

    dx = pts[:, 0] - x0
    dy = pts[:, 1] - y0
    distances = np.abs(-vy * dx + vx * dy)
    orthogonal_residual = float(np.mean(distances))

    # Variance along principal axis vs orthogonal variance
    proj = vx * dx + vy * dy
    var_proj = np.var(proj)
    var_ortho = np.var(distances)
    total_var = var_proj + var_ortho

    r_squared = float(var_proj / max(1e-6, total_var))
    angle_deg = float(np.degrees(np.arctan2(abs(vy), max(1e-5, abs(vx)))))

    return r_squared, angle_deg, orthogonal_residual


def filter_specular_lines(
    edge_map: np.ndarray,
    tip_x: int = 150,
    tip_y: int = 150,
    max_linearity_residual: float = 1.6,
    min_line_len: int = 12
) -> tuple[np.ndarray, np.ndarray]:
    """
    Erases straight horizontal lines, straight vertical lines, and angle-connected
    tool corners while strictly protecting active curved and helical chips.

    Parameters
    ----------
    edge_map : np.ndarray
        Binary edge map from Layer 4b.
    tip_x : int
        Tool tip x coordinate in ROI crop.
    tip_y : int
        Tool tip y coordinate in ROI crop.
    max_linearity_residual : float
        Maximum average orthogonal distance for a straight reflection streak. Default 1.6 px.
    min_line_len : int
        Minimum contour pixel length to evaluate. Default 12 px.

    Returns
    -------
    filtered_chip_edges : np.ndarray
        Clean edge map with specular reflection lines removed.
    removed_specular_map : np.ndarray
        Binary map of removed specular reflection artifacts for QA debugging.
    """
    filtered_chip_edges = edge_map.copy()
    removed_specular_map = np.zeros_like(edge_map)

    contours, _ = cv2.findContours(edge_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Fix 2.1: Asymmetric shear root & curl protection envelope (confined to immediate cutting nose)
    protect_x_min = tip_x - 25
    protect_x_max = tip_x + 25
    protect_y_min = tip_y - 25
    protect_y_max = tip_y + 15

    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        n_pts = len(pts)

        if n_pts < min_line_len:
            continue

        # Check if component overlaps the Fix 2.1 shear root protection zone
        xs, ys = pts[:, 0], pts[:, 1]
        mean_x, mean_y = float(np.mean(xs)), float(np.mean(ys))
        min_x, max_x = int(np.min(xs)), int(np.max(xs))
        min_y, max_y = int(np.min(ys)), int(np.max(ys))

        # If component's centroid or bounding box is inside the shear root envelope, protect it
        in_shear_box = (
            (protect_x_min <= mean_x <= protect_x_max and protect_y_min <= mean_y <= protect_y_max)
            or (min_x <= tip_x <= max_x and min_y <= tip_y <= max_y)
        )
        if in_shear_box:
            continue

        # Distance from component to cutting tip
        dists_to_tip = np.sqrt((xs - tip_x)**2 + (ys - tip_y)**2)
        min_dist_to_tip = float(np.min(dists_to_tip))

        is_removed = False

        # 1. Polygonal Corner & Tool Insert Corner Filtering (approxPolyDP)
        approx = cv2.approxPolyDP(cnt, epsilon=2.0, closed=False)
        if is_rigid_tool_corner(approx, min_segment_len=10.0):
            is_removed = True

        # 2. Straight Line Specular Streak Filter (Orthogonal Residual & R^2)
        if not is_removed:
            r_sq, angle_deg, residual = compute_linearity_r_squared(pts)

            # (a) General Horizontal / Vertical Glare Line
            if residual <= max_linearity_residual:
                is_horizontal = angle_deg <= 24.0  # within 24° of horizontal
                is_vertical = angle_deg >= 70.0    # within 20° of vertical

                if is_horizontal or is_vertical:
                    is_removed = True

            # (b) Fix 2.2: Cylindrical Reflection Bands across upper workpiece zone
            # Reflection bands form across y in [0, tip_y - 15] with high linearity (R^2 > 0.92)
            if not is_removed and mean_y <= (tip_y - 15):
                if r_sq >= 0.92 and angle_deg <= 16.0 and min_dist_to_tip >= 20.0:
                    is_removed = True

        if is_removed:
            cv2.drawContours(filtered_chip_edges, [cnt], -1, 0, thickness=cv2.FILLED)
            cv2.drawContours(removed_specular_map, [cnt], -1, 255, thickness=1)

    return filtered_chip_edges, removed_specular_map
