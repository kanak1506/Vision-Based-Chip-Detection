# -*- coding: utf-8 -*-
"""
Phase 3 Pipeline Forwarder
==========================
Points directly to the active Phase 3 v2 pipeline runner (scripts/run_phase3_v2_pipeline.py).
For the baseline v1 7-layer pipeline, see scripts/run_phase3_v1_baseline.py.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_phase3_v2_pipeline import run_phase3_v2_batch

if __name__ == "__main__":
    run_phase3_v2_batch()

