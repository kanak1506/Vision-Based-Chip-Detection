# -*- coding: utf-8 -*-
"""
Phase 0: Standalone Video Frame Extraction Runner
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.frame_extractor import run_frame_extraction

if __name__ == "__main__":
    run_frame_extraction()
