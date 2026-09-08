# -*- coding: utf-8 -*-
"""
=============================================================
  Phase 0: Stratified Video-Wise Metadata Dataset Splitting
  Vision-Based Metal Chip Classification & Machining Monitoring
=============================================================
"""

import os
import json
import random
import itertools
import pandas as pd
import numpy as np
from src.config import (
    VIDEO_METADATA_PATH,
    SPLITS_VIDEO_WISE_PATH,
    SPLITS_SUMMARY_PATH,
    RANDOM_SEED,
    SPLIT_RATIOS
)

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def get_material_from_index(video_idx: int) -> str:
    """Maps video index (1-27) to workpiece material category."""
    if video_idx <= 7:
        return "Aluminium"
    elif video_idx <= 13:
        return "Copper"
    else:
        return "Mild Steel"


def optimize_video_split(videos_df: pd.DataFrame, target_ratios=SPLIT_RATIOS, n_splits=(4, 1, 1)):
    """
    Selects the optimal whole-video partition to closely match target frame ratios
    while preventing any environmental or temporal background data leakage.
    """
    v_list = videos_df.to_dict('records')
    total_frames = sum(v['Total_Frames'] for v in v_list)
    target_train = total_frames * target_ratios[0]
    target_val   = total_frames * target_ratios[1]
    target_test  = total_frames * target_ratios[2]

    best_assignment = None
    best_error = float('inf')

    n_total = len(v_list)
    n_train, n_val, n_test = n_splits
    indices = list(range(n_total))

    for val_indices in itertools.combinations(indices, n_val):
        rem1 = [i for i in indices if i not in val_indices]
        for test_indices in itertools.combinations(rem1, n_test):
            train_indices = [i for i in rem1 if i not in test_indices]

            train_f = sum(v_list[i]['Total_Frames'] for i in train_indices)
            val_f   = sum(v_list[i]['Total_Frames'] for i in val_indices)
            test_f  = sum(v_list[i]['Total_Frames'] for i in test_indices)

            err = (
                abs(train_f - target_train) ** 2 +
                abs(val_f - target_val) ** 2 +
                abs(test_f - target_test) ** 2
            )

            if err < best_error:
                best_error = err
                best_assignment = {
                    'train': [v_list[i]['Filename'] for i in train_indices],
                    'val':   [v_list[i]['Filename'] for i in val_indices],
                    'test':  [v_list[i]['Filename'] for i in test_indices],
                }

    return best_assignment


def run_splitting_pipeline(
    video_meta_path: str = VIDEO_METADATA_PATH,
    output_video_split: str = SPLITS_VIDEO_WISE_PATH,
    output_summary_json: str = SPLITS_SUMMARY_PATH
) -> dict:
    """Executes stratified video-wise metadata splitting."""
    print("=" * 70)
    print("  PHASE 0: STRATIFIED VIDEO-WISE DATASET SPLITTING (METADATA)")
    print("=" * 70)

    # 1. Load Video Metadata
    df_vm = pd.read_csv(video_meta_path)
    df_vm["Material"] = df_vm["Index"].apply(get_material_from_index)

    print(f"Loaded {len(df_vm)} videos across 3 materials:")
    print(df_vm.groupby("Material")["Total_Frames"].agg(["count", "sum"]).rename(columns={"count": "Videos", "sum": "Frames"}))

    # 2. Optimized Video-Wise Split Plan
    video_assignments = {}
    material_split_plans = {
        "Aluminium":  (5, 1, 1),
        "Copper":     (4, 1, 1),
        "Mild Steel": (10, 2, 2)
    }

    for mat, n_splits in material_split_plans.items():
        mat_vids = df_vm[df_vm["Material"] == mat]
        assignment = optimize_video_split(mat_vids, target_ratios=SPLIT_RATIOS, n_splits=n_splits)
        video_assignments[mat] = assignment

    # Map split to DataFrame
    vid_to_split = {}
    for mat, splits in video_assignments.items():
        for split_name, vlist in splits.items():
            for v in vlist:
                vid_to_split[v] = split_name

    df_vm["Split"] = df_vm["Filename"].map(vid_to_split)

    # 3. Save Output Manifest
    os.makedirs(os.path.dirname(output_video_split), exist_ok=True)
    df_vm.to_csv(output_video_split, index=False)

    # 4. Compile Summary Statistics
    summary = {
        "dataset_statistics": {
            "total_videos": len(df_vm),
            "total_raw_frames": int(df_vm["Total_Frames"].sum()),
            "total_duration_sec": round(float(df_vm["Duration_Sec"].sum()), 2)
        },
        "target_ratios": {"train": SPLIT_RATIOS[0], "val": SPLIT_RATIOS[1], "test": SPLIT_RATIOS[2]},
        "video_assignments": video_assignments,
        "distribution_summary": {
            split: {
                "videos": int((df_vm["Split"] == split).sum()),
                "frames": int(df_vm[df_vm["Split"] == split]["Total_Frames"].sum()),
                "percentage": round(100.0 * df_vm[df_vm["Split"] == split]["Total_Frames"].sum() / df_vm["Total_Frames"].sum(), 1)
            }
            for split in ["train", "val", "test"]
        }
    }

    with open(output_summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "-" * 70)
    print("  SPLIT DISTRIBUTION BREAKDOWN:")
    for split, data in summary["distribution_summary"].items():
        print(f"    - {split.upper():5s}: {data['videos']} videos | {data['frames']:,} frames ({data['percentage']}%)")
    print("-" * 70)
    print(f"\nSaved Video-Wise Split Manifest: '{output_video_split}'")
    print(f"Saved Split Summary JSON:        '{output_summary_json}'")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    run_splitting_pipeline()
