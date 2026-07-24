"""
Mask quality filtering module: filter empty masks to improve 3DGS segmentation quality.
"""
import os
import cv2
import numpy as np
from tqdm import tqdm


def filter_low_quality_masks(mask_folder, empty_threshold=0.001):
    """
    Filter empty masks (foreground pixel ratio below threshold).

    Parameters:
        mask_folder: mask folder path (required)
        empty_threshold: empty mask threshold (if the pixel ratio is less than this value, it is considered empty)

    Returns:
        removed_indices: list of removed frame indices
    """

    if not os.path.exists(mask_folder):
        raise FileNotFoundError(f"Mask folder does not exist: {mask_folder}")

    # Get all mask files
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith(('.jpg', '.png'))])

    if len(mask_files) == 0:
        print("Warning: No mask files found")
        return []

    removed_indices = []

    for mask_file in tqdm(mask_files, desc="Empty mask filtering"):
        mask_path = os.path.join(mask_folder, mask_file)
        frame_idx = int(os.path.splitext(mask_file)[0])

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask_sum = np.sum(mask > 128)
        if mask_sum / mask.size <= empty_threshold:
            removed_indices.append(frame_idx)
            os.remove(mask_path)

    if removed_indices:
        print(f"Removed {len(removed_indices)} empty masks")

    return removed_indices
