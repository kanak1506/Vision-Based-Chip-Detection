# -*- coding: utf-8 -*-
"""
Module: sector_masker.py
Step 3.5: Asymmetric Spatial Cutting Sector Masking & Workpiece Glare Suppression.
Isolates the active machining shear zone around the tool tip while suppressing workpiece
cylinder glare and lathe bed floor clutter (Filep et al. method).
"""

import cv2
import numpy as np


def create_cutting_sector_mask(
    shape: tuple = (300, 300),
    tip_x: int = 150,
    tip_y: int = 150,
    cutting_radius: int = 125
) -> np.ndarray:
    """
    Constructs an exact physical machining sector mask:
      - Retains circular cutting proximity corridor around (tip_x, tip_y).
      - Suppresses solid workpiece cylinder body (x <= tip_x - 5, y <= tip_y - 12).
      - Suppresses lathe chuck / spindle body (x >= tip_x + 35, y <= tip_y - 15).
      - Suppresses far background top (y <= 55).
      - [Fix 1.1] Floor/shank suppression: y >= tip_y + 35 across x <= tip_x + 35
          (closes the shank gap between the old left-floor rect and the carriage rect
          that was leaking floor debris through the x=[tip_x-28, tip_x+30] corridor).
      - Suppresses far left floor (x <= tip_x - 55, y >= tip_y + 15).
      - Suppresses tool block & carriage (x >= tip_x + 30, y >= tip_y + 15).
      - Unmasks 100% of the Universal Machining Envelope (applied last; always wins):
          * [Fix 1.2] Rightward Ribbon Flow: x in [tip_x-10, tip_x+150], y in [tip_y-60, tip_y+15]
            (confined to air zone at/above cutting height; preserves rightward ribbon while keeping carriage suppressed below tip_y+15)
          * Downward/Leftward Helical Flow: x in [tip_x-120, tip_x+10], y in [tip_y-10, tip_y+130]
            (expanded from old [tip_x-28, tip_x+10]/[tip_y-10, tip_y+35] — measured from
             actual helical coil frame: coil body spans x~tip_x-120..tip_x-30, y~tip_y+10..tip_y+130)
       - Hard-masks tool shank body (x in [tip_x-10, tip_x+40], y >= tip_y+25) applied LAST
         to override any corridor unmasking and eliminate the vertical shank edge false positive.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # 1. Active circular proximity around tool tip
    cv2.circle(mask, (int(tip_x), int(tip_y)), cutting_radius, 255, -1)

    # 2. Suppress solid workpiece cylinder body
    cv2.rectangle(mask, (0, 0), (int(tip_x - 5), int(tip_y - 12)), 0, -1)

    # 3. Suppress lathe chuck & spindle body
    cv2.rectangle(mask, (int(tip_x + 35), 0), (w, int(tip_y - 15)), 0, -1)

    # 4. Suppress far upper machine bed (y <= 55)
    cv2.rectangle(mask, (0, 0), (w, 55), 0, -1)

    # 5. [Fix 1.1] Floor/shank suppression: y >= tip_y+35, x <= tip_x+35
    #    Replaces the old narrow left-floor rect (x <= tip_x-28) with a wider
    #    rectangle that also covers the shank corridor up to x = tip_x+35.
    cv2.rectangle(mask, (0, int(tip_y + 35)), (int(tip_x + 35), h), 0, -1)

    # 6. Suppress far left floor
    cv2.rectangle(mask, (0, int(tip_y + 15)), (int(tip_x - 55), h), 0, -1)

    # 7. Tool block & carriage suppression (x >= tip_x + 30, y >= tip_y + 15) REMOVED per user request

    # 8. Unmask rightward ribbon flow corridor — air zone [tip_y - 60, tip_y + 90]
    cv2.rectangle(mask,
                  (int(tip_x - 10),  int(tip_y - 60)),
                  (int(tip_x + 150), int(tip_y + 90)),
                  255, -1)

    # 9. Unmask downward/leftward helical flow corridor
    cv2.rectangle(mask,
                  (int(tip_x - 120), int(tip_y - 10)),
                  (int(tip_x + 10),  int(tip_y + 130)),
                  255, -1)

    # 10. Hard-mask tool shank body (applied LAST — overrides all corridor unmasking).
    #     The vertical metal shank below the carbide insert causes a strong step edge.
    #     Starts at tip_y + 25 to preserve the active cutting nose/flank while blocking the shank.
    cv2.rectangle(mask,
                  (int(tip_x - 10), int(tip_y + 25)),
                  (int(tip_x + 40), h),
                  0, -1)

    return mask


def apply_sector_mask(
    edge_map: np.ndarray,
    tip_x: int = 150,
    tip_y: int = 150,
    cutting_radius: int = 140
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies asymmetric cutting sector mask to an edge map.
    Returns (masked_edges, sector_mask).
    """
    mask = create_cutting_sector_mask(edge_map.shape, tip_x, tip_y, cutting_radius)
    masked_edges = cv2.bitwise_and(edge_map, edge_map, mask=mask)
    return masked_edges, mask
