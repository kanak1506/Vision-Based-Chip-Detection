# -*- coding: utf-8 -*-
"""
Phase 0: Run Video Metadata Extraction and Stratified Splitting
"""

import sys
import os

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metadata_extractor import extract_video_metadata
from src.dataset_splitter import run_splitting_pipeline

if __name__ == "__main__":
    print("=== Step 1: Auditing Videos & Extracting Metadata ===")
    extract_video_metadata()

    print("\n=== Step 2: Running Stratified Dataset Splitting ===")
    run_splitting_pipeline()
