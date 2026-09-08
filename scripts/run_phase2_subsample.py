# -*- coding: utf-8 -*-
"""
Phase 2: Run Frame Subsampling (~4 FPS) & ROI Crop Extraction
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.subsample_filter import run_subsampling_pipeline

if __name__ == "__main__":
    run_subsampling_pipeline()
