# -*- coding: utf-8 -*-
"""
test_specular_prefilter.py
Unit test for Layer 0: Dichromatic Specular Pre-Filter

Tests:
1. Synthetic glare patch - mask coverage and pixel repair
2. Zero-specular image  - no mask pixels flagged
3. Full-image glare     - graceful edge-case handling
4. Real crop scan       - confirm specular pixels exist in the dataset
                          (Image-4-equivalent cylinder-reflection check)
                          Uses mask-only scan (cheap), inpaints only the winner.
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from src.config import PROJECT_ROOT, PHASE2_CROPS_DIR
from src.specular_prefilter import (
    apply_specular_prefilter,
    build_specular_mask,
    inpaint_specular,
    DEFAULT_S_THRESH,
    DEFAULT_V_THRESH,
)

PHASE2_CROPS = str(PHASE2_CROPS_DIR / "retained")
OUT_DIR = str(PROJECT_ROOT / "scratch" / "test_layer0_specular")
os.makedirs(OUT_DIR, exist_ok=True)

PASS = "[PASS]"
FAIL = "[FAIL]"
_results = []


def report(name, ok, detail=""):
    tag = PASS if ok else FAIL
    msg = f"{tag} {name}"
    if detail:
        msg += f"  — {detail}"
    print(msg)
    _results.append(ok)


# -----------------------------------------------------------------------
# Test 1: Synthetic glare patch
# -----------------------------------------------------------------------
def test_synthetic_glare():
    h, w = 300, 300
    img = np.full((h, w, 3), fill_value=[60, 100, 80], dtype=np.uint8)
    # Inject bright, low-saturation patch: BGR (240, 242, 244) -> HSV low-S/high-V
    img[80:120, 100:200] = [240, 242, 244]

    inpainted, mask, pct = apply_specular_prefilter(img)

    patch_flagged = int(np.count_nonzero(mask[80:120, 100:200]))
    patch_total   = 40 * 100
    ok_mask = patch_flagged > patch_total * 0.5
    report("Synthetic glare — mask covers injected patch",
           ok_mask, f"{patch_flagged}/{patch_total} px flagged ({pct:.2f}% total)")

    diff_inside = np.mean(
        np.abs(inpainted[mask == 255].astype(int) - img[mask == 255].astype(int))
    )
    ok_inpaint = diff_inside > 0.5
    report("Synthetic glare — inpainting modified flagged pixels",
           ok_inpaint, f"mean delta inside mask: {diff_inside:.2f}")

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(OUT_DIR, "synthetic_qa.png"), np.hstack([img, mask_bgr, inpainted]))


# -----------------------------------------------------------------------
# Test 2: Zero-specular image (dark, saturated)
# -----------------------------------------------------------------------
def test_no_glare():
    img_hsv      = np.zeros((100, 100, 3), dtype=np.uint8)
    img_hsv[:]   = [30, 180, 100]          # saturated mid-brightness
    img          = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    _, mask, pct = apply_specular_prefilter(img)
    ok = pct == 0.0
    report("No-glare image — zero specular pixels flagged", ok, f"specular_pct={pct:.4f}%")


# -----------------------------------------------------------------------
# Test 3: Full-image glare (edge case)
# -----------------------------------------------------------------------
def test_full_glare():
    img_hsv          = np.zeros((100, 100, 3), dtype=np.uint8)
    img_hsv[:, :, 2] = 255    # S=0, V=255 => all pixels specular
    img              = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    inpainted, mask, pct = apply_specular_prefilter(img)
    ok = pct == 100.0 and inpainted is not None and inpainted.shape == img.shape
    report("Full-glare edge case — handled without error", ok, f"specular_pct={pct:.1f}%")


# -----------------------------------------------------------------------
# Test 4: Real crop — find highest-specular frame (Image-4 equivalent)
#   Scan uses build_specular_mask only (no inpaint) for speed.
#   Inpaint is called only on the single winner.
# -----------------------------------------------------------------------
def test_real_crop_highest_glare():
    if not os.path.isdir(PHASE2_CROPS):
        report("Real crop — phase2 crops directory found", False,
               f"not found: {PHASE2_CROPS}")
        return

    best_pct  = 0.0
    best_path = None
    best_img  = None
    best_mask = None
    scanned   = 0

    for root, _, files in os.walk(PHASE2_CROPS):
        for fname in sorted(files):
            if not fname.lower().endswith((".jpg", ".png")):
                continue
            fpath = os.path.join(root, fname)
            img   = cv2.imread(fpath)
            if img is None:
                continue
            mask, pct = build_specular_mask(img)   # mask-only, no inpaint
            scanned  += 1
            if pct > best_pct:
                best_pct  = pct
                best_path = fpath
                best_img  = img
                best_mask = mask

    print(f"  Scanned {scanned} crops.")
    ok = best_pct > 0.0
    report("Real crop — specular pixels found in dataset",
           ok, f"best {best_pct:.2f}% — {os.path.basename(best_path) if best_path else 'N/A'}")

    if best_img is not None and np.any(best_mask):
        # Now inpaint only the winner
        inpainted = inpaint_specular(best_img, best_mask)
        diff_inside = np.mean(
            np.abs(inpainted[best_mask == 255].astype(int)
                   - best_img[best_mask == 255].astype(int))
        )
        ok2 = diff_inside > 0.0
        report("Real crop — inpainting modified specular pixels", ok2,
               f"mean delta inside mask: {diff_inside:.2f}")

        mask_bgr = cv2.cvtColor(best_mask, cv2.COLOR_GRAY2BGR)
        out_path = os.path.join(OUT_DIR, "real_crop_highest_glare.png")
        cv2.imwrite(out_path, np.hstack([best_img, mask_bgr, inpainted]))
        print(f"  QA image saved -> {out_path}")


# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 62)
    print("Layer 0 Unit Test — Dichromatic Specular Pre-Filter")
    print(f"  S_thresh={DEFAULT_S_THRESH}, V_thresh={DEFAULT_V_THRESH}")
    print("=" * 62)

    test_synthetic_glare()
    test_no_glare()
    test_full_glare()
    test_real_crop_highest_glare()

    print("=" * 62)
    total  = len(_results)
    passed = sum(_results)
    print(f"Result: {passed}/{total} tests passed")
    sys.exit(0 if passed == total else 1)
