# -*- coding: utf-8 -*-
"""
Phase 1: Standalone Tracked Frames Rendering Runner
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tracking_renderer import run_phase1_frame_rendering

if __name__ == "__main__":
    run_phase1_frame_rendering()
