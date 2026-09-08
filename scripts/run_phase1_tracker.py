# -*- coding: utf-8 -*-
"""
Phase 1: Run Tool Tip Tracking & Dynamic Trajectory Logging
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.robust_tool_tracker import run_batch_robust_tracking

if __name__ == "__main__":
    run_batch_robust_tracking()
