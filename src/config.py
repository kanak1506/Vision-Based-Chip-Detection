# -*- coding: utf-8 -*-
"""
=============================================================
  Centralized Configuration & Path Management
  Vision-Based Metal Chip Classification & Machining Monitoring
=============================================================
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEOS_DIR = DATA_DIR / "phase0_raw_videos"
METADATA_DIR = DATA_DIR / "metadata"

# Phase Datasets
EXTRACTED_FRAMES_DIR = DATA_DIR / "phase0_extracted_frames"
PHASE1_TRACKED_DIR = DATA_DIR / "phase1_tracked_frames"
PHASE2_CROPS_DIR = DATA_DIR / "phase2_roi_crops"
# Phase 3 Stepwise Datasets
PHASE3_STEPWISE_DIR = DATA_DIR / "phase3_stepwise_layers"
PHASE3_L0_SPECULAR_DIR = PHASE3_STEPWISE_DIR / "layer0_specular_prefiltered"
PHASE3_L1_LAB_DIR = PHASE3_STEPWISE_DIR / "layer1_lab_denoised"
PHASE3_L2_FRANGI_DIR = PHASE3_STEPWISE_DIR / "layer2_frangi_vesselness"
PHASE3_L2B_FRANGI_DIR = PHASE3_L2_FRANGI_DIR
PHASE3_L2C_COMBINED_DIR = PHASE3_L2_FRANGI_DIR
PHASE3_L3_SECTOR_DIR = PHASE3_STEPWISE_DIR / "layer3_sector_masked"
PHASE3_L4_ADAPTIVE_BG_DIR = PHASE3_STEPWISE_DIR / "layer4_adaptive_background"
PHASE3_L4B_FLOW_DIR = PHASE3_STEPWISE_DIR / "layer4b_optical_flow_gated"
PHASE3_L5_SPECULAR_DIR = PHASE3_STEPWISE_DIR / "layer5_specular_filtered"
PHASE3_L6_SEED_CONNECTED_DIR = PHASE3_STEPWISE_DIR / "layer6_seed_connected_chips"
PHASE3_L7_SKELETON_DIR = PHASE3_STEPWISE_DIR / "layer7_morphological_skeleton"
PHASE3_L8_OVERLAYS_DIR = PHASE3_STEPWISE_DIR / "layer8_chip_overlays"

# Backwards Compatibility Aliases
PHASE3_L4_SUBTRACTED_DIR = PHASE3_L4_ADAPTIVE_BG_DIR
PHASE3_L6_CONTOURS_DIR = PHASE3_L6_SEED_CONNECTED_DIR
PHASE3_DENOISED_DIR = PHASE3_L1_LAB_DIR
PHASE3_EDGES_DIR = PHASE3_L2C_COMBINED_DIR
PHASE3_SUBTRACTED_DIR = PHASE3_L4_ADAPTIVE_BG_DIR
PHASE3_SECTOR_DIR = PHASE3_L3_SECTOR_DIR
PHASE3_CONTOURS_DIR = PHASE3_L6_SEED_CONNECTED_DIR
PHASE3_OVERLAYS_DIR = PHASE3_L8_OVERLAYS_DIR

# Metadata Artifacts
VIDEO_METADATA_PATH = METADATA_DIR / "video_metadata.csv"
SPLITS_VIDEO_WISE_PATH = METADATA_DIR / "dataset_splits_video_wise.csv"
SPLITS_SUMMARY_PATH = METADATA_DIR / "dataset_splits_summary.json"
TRACKER_CONFIG_PATH = METADATA_DIR / "tracker_config.json"
TRACKING_RESULTS_PATH = METADATA_DIR / "tracking_results.csv"
PHASE2_MANIFEST_PATH = METADATA_DIR / "phase2_subsample_manifest.csv"
PHASE2_SUMMARY_PATH = METADATA_DIR / "phase2_summary.json"
PHASE3_MANIFEST_PATH = METADATA_DIR / "phase3_manifest.csv"

# Project Folders
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_UNIVERSAL_DIR = RESULTS_DIR / "universal_solution"
DOCS_DIR = PROJECT_ROOT / "docs"

# Pipeline Parameters
SAMPLE_EVERY_N = 15          # ~4 FPS sampling rate on 60 FPS video
TOOL_MATCH_THRESHOLD = 0.45  # Normalized Cross-Correlation tool verification threshold
MAX_TOOL_STEP_DISPLACEMENT = 80  # Max allowable displacement between sampled frames along feed axis
CROP_SCALE = 3.5             # ROI crop scale factor around tool bbox
MIN_CROP_HALF = 150          # Ensures minimum 300x300px ROI window
RANDOM_SEED = 42             # Deterministic seed for reproducible splits
N_WORKERS = min(8, os.cpu_count() or 4)

SPLIT_RATIOS = (0.70, 0.15, 0.15)  # Train / Val / Test target partition ratios


def sanitize_filename(fname: str) -> str:
    """Sanitizes video or image filename by stripping extension and replacing spaces/colons."""
    base = os.path.splitext(fname)[0]
    return base.replace(" ", "_").replace(":", "-")


def draw_tool_tip_blue_dot(image, x: int, y: int, radius: int = 6, with_label: bool = False):
    """
    Standardized visualization helper:
    Tool-tip blue dot disabled per user preference. Coordinates remain in metadata.
    """
    return image


def to_relative_path(path) -> str:
    """
    Converts an absolute or relative path to a POSIX-style path relative to PROJECT_ROOT.
    Ensures saved manifests and summaries are portable across operating systems and environments.
    """
    p = Path(path)
    if p.is_absolute():
        try:
            p = p.relative_to(PROJECT_ROOT)
        except ValueError:
            pass
    return p.as_posix()


