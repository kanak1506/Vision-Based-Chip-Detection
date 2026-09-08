# -*- coding: utf-8 -*-
"""
Module: contour_extractor.py
Step 3.8: Cutting Zone Proximity Clustering & Clean Contour Extraction (Filep et al. Step e).
Extracts connected dynamic chip contours originating within the cutting zone proximity (<= 160 px)
from the cutting tool tip and filters out minor noise speckles.
"""

import cv2
import numpy as np


def link_edge_endpoints(
    edge_map: np.ndarray,
    max_link_dist: float = 6.0,
    min_segment_len: int = 8
) -> np.ndarray:
    """
    Bridges micro-gaps between adjacent open Canny edge endpoints along real chip ribbons
    and coils without creating 2D blobs.
    """
    linked_map = edge_map.copy()
    contours, _ = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    valid = [c for c in contours if len(c) >= min_segment_len]
    if len(valid) < 2:
        return linked_map

    endpoints = []
    for cnt in valid:
        pts = cnt.reshape(-1, 2)
        endpoints.append((pts[0], pts[-1]))

    n = len(endpoints)
    for i in range(n):
        p1_start, p1_end = endpoints[i]
        for j in range(i + 1, n):
            p2_start, p2_end = endpoints[j]

            # Check pairwise Euclidean distances between terminal endpoints
            dists = [
                (float(np.linalg.norm(p1_start - p2_start)), p1_start, p2_start),
                (float(np.linalg.norm(p1_start - p2_end)), p1_start, p2_end),
                (float(np.linalg.norm(p1_end - p2_start)), p1_end, p2_start),
                (float(np.linalg.norm(p1_end - p2_end)), p1_end, p2_end)
            ]
            min_d, ptA, ptB = min(dists, key=lambda x: x[0])
            if min_d <= max_link_dist:
                cv2.line(linked_map, (int(ptA[0]), int(ptA[1])), (int(ptB[0]), int(ptB[1])), 255, 1, lineType=cv2.LINE_4)

    return linked_map


def filter_seed_connected_components(
    edge_map: np.ndarray,
    tip_x: int = 150,
    tip_y: int = 150,
    seed_radius: int = 25,
    max_gap_helical: float = 22.0,
    max_gap_ribbon: float = 40.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Layer 6 (Phase 3 v2): Tool-Tip Seed-Connected Component Tracing.
    Retains edge components physically originating at the tool tip shear root
    or forming continuous stream continuations along valid chip ejection corridors:
      - Rightward Ribbon Corridor: x in [tip_x - 10, tip_x + 150], y in [tip_y - 60, tip_y + 90]
      - Downward Helical Corridor: x in [tip_x - 65, tip_x + 20], y in [tip_y - 15, tip_y + 95]

    Excludes invariant non-machining zones:
      - Lower-Left Static Bed Clutter: x < tip_x - 20 and y > tip_y + 20
      - Lower-Right Steel Tool Post Body: x > tip_x + 30 and y > tip_y + 20
      - Upper Workpiece Surface: x < tip_x - 30 and y < tip_y - 25
      - Tool Shank Body: x in [tip_x - 10, tip_x + 40] and y > tip_y + 12

    Parameters
    ----------
    edge_map : np.ndarray
        Binary edge map from Layer 5 (specular filtered).
    tip_x : int
        Tool tip x coordinate in ROI crop. Default 150.
    tip_y : int
        Tool tip y coordinate in ROI crop. Default 150.
    seed_radius : int
        Radius around tool tip defining the initial shear initiation seed. Default 25.
    max_gap_helical : float
        Maximum allowable gap between descending helical coil loops. Default 22.0 px.
    max_gap_ribbon : float
        Maximum allowable gap along continuous horizontal ribbons. Default 40.0 px.

    Returns
    -------
    seed_connected_edges : np.ndarray
        Clean edge map containing only chip edges connected to the cutting root.
    discarded_clutter_map : np.ndarray
        Binary map of rejected disconnected clutter for QA.
    """
    h, w = edge_map.shape[:2]
    if cv2.countNonZero(edge_map) == 0:
        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)

    # 1. Invariant Machine Sector Filtering
    valid_sector_mask = np.ones((h, w), dtype=np.uint8)
    # Block lower-left lathe floor clutter
    valid_sector_mask[int(tip_y + 20):, :int(tip_x - 20)] = 0
    # Block upper workpiece glare line
    valid_sector_mask[:int(tip_y - 25), :int(tip_x - 30)] = 0
    # Block tool shank body below insert (starts at tip_y + 25 to preserve insert flank)
    valid_sector_mask[int(tip_y + 25):, int(tip_x - 10):int(tip_x + 40)] = 0

    cleaned_input = cv2.bitwise_and(edge_map, edge_map, mask=valid_sector_mask)

    # 2. Connected Components Analysis on 8-connectivity
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_input, connectivity=8)
    if num_labels <= 1:
        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)

    # 3. Initial Seed Mask & Immediate Shear Proximity at Cutting Tool Tip
    seed_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(seed_mask, (int(tip_x), int(tip_y)), seed_radius, 255, -1)

    seed_connected_labels = set()
    active_tree_mask = np.zeros((h, w), dtype=np.uint8)

    for lbl in range(1, num_labels):
        comp_mask = (labels == lbl).astype(np.uint8)
        cx, cy = centroids[lbl]
        dist_to_tip = float(np.sqrt((cx - tip_x)**2 + (cy - tip_y)**2))
        # Retain if component touches seed circle or sits in immediate shear proximity (<= 25px)
        if cv2.countNonZero(cv2.bitwise_and(comp_mask, seed_mask)) > 0 or dist_to_tip <= 25.0:
            seed_connected_labels.add(lbl)
            active_tree_mask[labels == lbl] = 255

    # 4. Multi-Hop Geodesic Stream Propagation
    # Chains connectivity down the stream (C0 -> C1 -> C2 -> C3...) along active ejection corridors
    if seed_connected_labels:
        ribbon_x_min, ribbon_x_max = tip_x - 10, tip_x + 150
        ribbon_y_min, ribbon_y_max = tip_y - 60, tip_y + 90

        helical_x_min, helical_x_max = tip_x - 65, tip_x + 20
        helical_y_min, helical_y_max = tip_y - 15, tip_y + 95

        changed = True
        while changed:
            changed = False
            dist_map = cv2.distanceTransform(cv2.bitwise_not(active_tree_mask), cv2.DIST_L2, 3)

            for lbl in range(1, num_labels):
                if lbl in seed_connected_labels:
                    continue

                cx, cy = centroids[lbl]
                area = stats[lbl, cv2.CC_STAT_AREA]
                if area < 5:
                    continue

                in_ribbon = (ribbon_x_min <= cx <= ribbon_x_max and ribbon_y_min <= cy <= ribbon_y_max)
                in_helical = (helical_x_min <= cx <= helical_x_max and helical_y_min <= cy <= helical_y_max)

                if in_ribbon or in_helical:
                    comp_dist = float(np.min(dist_map[labels == lbl]))
                    allowed_gap = max_gap_helical if in_helical else max_gap_ribbon
                    if comp_dist <= allowed_gap:
                        seed_connected_labels.add(lbl)
                        active_tree_mask[labels == lbl] = 255
                        changed = True

    # 5. Construct Clean Output Edge Map & Discarded Clutter Map
    seed_connected_edges = np.zeros((h, w), dtype=np.uint8)
    discarded_clutter_map = np.zeros((h, w), dtype=np.uint8)

    for lbl in range(1, num_labels):
        if lbl in seed_connected_labels:
            seed_connected_edges[labels == lbl] = 255
        else:
            discarded_clutter_map[labels == lbl] = 255

    return seed_connected_edges, discarded_clutter_map


def orientation_adaptive_closing(
    edge_map: np.ndarray,
    kernel_len: int = 5
) -> np.ndarray:
    """
    Applies multi-directional morphological closing along local tangent orientations:
      angles = [0°, 30°, 45°, 60°, 90°, 120°, 135°, 150°]
    Bridges micro-gaps along curving chip ribbons and helical coils without bloating into 2D blobs.
    """
    if cv2.countNonZero(edge_map) == 0:
        return edge_map.copy()

    angles = [0, 30, 45, 60, 90, 120, 135, 150]
    closed_union = np.zeros_like(edge_map)

    for deg in angles:
        rad = np.deg2rad(deg)
        k_size = kernel_len
        kernel = np.zeros((k_size, k_size), dtype=np.uint8)
        center = k_size // 2

        dx = int(round(np.cos(rad) * center))
        dy = int(round(np.sin(rad) * center))
        cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1, 1)

        closed = cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel)
        closed_union = cv2.bitwise_or(closed_union, closed)

    return closed_union


def extract_chip_contours(
    edge_map: np.ndarray,
    tip_x: int = 150,
    tip_y: int = 150,
    max_dist_to_tip: float = 140.0,
    min_contour_length: int = 4,
    max_link_dist: float = 6.0,
    filter_seed: bool = False
) -> tuple[np.ndarray, list, dict]:
    """
    Layer 7 (Phase 3 v2): Filled Medial Axis Skeletonization & Morphological Metrics.
    Collapses blurred 2D ribbon strips into a single true 1-pixel central spine (no hollow border loops).

    Parameters
    ----------
    edge_map : np.ndarray
        Binary edge map from Layer 6 (seed-connected chips).
    tip_x : int
        Tool tip x coordinate in ROI crop. Default 150.
    tip_y : int
        Tool tip y coordinate in ROI crop. Default 150.
    max_dist_to_tip : float
        Maximum allowable distance from tool tip to chip contour. Default 140.0.
    min_contour_length : int
        Minimum contour point length. Default 4.
    max_link_dist : float
        Maximum distance to bridge between terminal endpoints. Default 6.0.
    filter_seed : bool
        Whether to re-apply seed filtering (default False).

    Returns
    -------
    skeleton_map : np.ndarray
        Crisp 1-pixel binary centerline spine of the segmented chip.
    valid_contours : list
        List of segmented OpenCV contours.
    metrics : dict
        Dict with keys: num_chip_clusters, total_chip_area, total_chip_perimeter, min_dist_to_tip.
    """
    h, w = edge_map.shape[:2]
    clean_skeleton_mask = np.zeros((h, w), dtype=np.uint8)

    if cv2.countNonZero(edge_map) == 0:
        return clean_skeleton_mask, [], {
            "num_chip_clusters": 0,
            "total_chip_area": 0.0,
            "total_chip_perimeter": 0.0,
            "min_dist_to_tip": 0.0
        }

    if filter_seed:
        active_edges, _ = filter_seed_connected_components(edge_map, tip_x=tip_x, tip_y=tip_y)
    else:
        active_edges = edge_map

    # 1. Orientation-Adaptive Closing (bridges reflection dips)
    closed = orientation_adaptive_closing(active_edges, kernel_len=5)

    # 2. Medial Axis Solid Fill (collapses dual boundaries into a solid strip before thinning)
    k_fill = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    filled_ribbon = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, k_fill)

    # 3. Zhang-Suen Topological Thinning into crisp 1px centerline spine
    try:
        thinned = cv2.ximgproc.thinning(filled_ribbon, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except Exception:
        thinned = filled_ribbon

    # 4. Endpoint Proximity Linking across micro-gaps
    if max_link_dist > 0:
        linked_edges = link_edge_endpoints(thinned, max_link_dist=max_link_dist, min_segment_len=6)
    else:
        linked_edges = thinned

    # 5. Extract continuous connected contours along the 1px centerline
    contours, _ = cv2.findContours(linked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    valid_contours = []
    total_area = 0.0
    total_perimeter = 0.0
    min_dist = float("inf")

    tip_pt = np.array([tip_x, tip_y], dtype=np.float32)

    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        n_pts = len(pts)

        if n_pts < min_contour_length:
            continue

        dists = np.linalg.norm(pts.astype(np.float32) - tip_pt, axis=1)
        cnt_min_dist = float(np.min(dists))

        if cnt_min_dist <= max_dist_to_tip:
            valid_contours.append(cnt)
            cv2.drawContours(clean_skeleton_mask, [cnt], -1, 255, thickness=1)

            area = cv2.contourArea(cnt)
            perim = cv2.arcLength(cnt, False)
            total_area += max(area, float(n_pts))
            total_perimeter += perim
            if cnt_min_dist < min_dist:
                min_dist = cnt_min_dist

    if min_dist == float("inf"):
        min_dist = 0.0

    metrics = {
        "num_chip_clusters": len(valid_contours),
        "total_chip_area": round(total_area, 2),
        "total_chip_perimeter": round(total_perimeter, 2),
        "min_dist_to_tip": round(min_dist, 2)
    }

    return clean_skeleton_mask, valid_contours, metrics


