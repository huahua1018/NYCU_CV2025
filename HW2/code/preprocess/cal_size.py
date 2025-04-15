"""
This script analyzes image dimensions in multiple dataset folders.
It collects width, height, aspect ratio, and counts the most common image sizes.
Useful for data resizing strategy.
"""

import os
from collections import Counter

from PIL import Image
from tqdm import tqdm
import numpy as np


def analyze_image_sizes(folder_path):
    """
    Analyze the dimensions of all images in a given folder.

    Args:
        folder_path (str): Path to the folder containing images.

    Returns:
        widths (List[int]): List of image widths.
        heights (List[int]): List of image heights.
        size_counter (dict): Dictionary counting occurrences of (width, height) pairs.
        ratio (List[float]): List of aspect ratios (shorter side / longer side).
    """
    widths, heights = [], []
    size_counter = {}
    ratio = []

    # Iterate over all image files in the folder
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)

                    # Count each (width, height) pair
                    size_counter[(w, h)] = size_counter.get((w, h), 0) + 1

                    # Calculate aspect ratio (shorter / longer)
                    ratio.append(min(w, h) / max(w, h))
            except Exception as e:
                print(f"Error loading image {filename}: {e}")

    return widths, heights, size_counter, ratio


# List of dataset folders to analyze
folders = [
    "../../nycu-hw2-data/train/",
    "../../nycu-hw2-data/valid/",
    "../../nycu-hw2-data/test/",
]

for f in folders:
    print(f"Folder: {f}")
    cal_widths, cal_heights, cal_size_counter, cal_ratio = analyze_image_sizes(f)

    print(f"Number of images: {len(cal_widths)}")
    print(f"Minimum size: ({min(cal_widths)}, {min(cal_heights)})")
    print(f"Maximum size: ({max(cal_widths)}, {max(cal_heights)})")
    print(f"Average size: ({np.mean(cal_widths):.2f}, {np.mean(cal_heights):.2f})")

    print(f"Minimum aspect ratio: {min(cal_ratio):.2f}")
    print(f"Maximum aspect ratio: {max(cal_ratio):.2f}")
    print(f"Average aspect ratio: {np.mean(cal_ratio):.2f}")

    # Print top 5 most common image sizes
    print("Top 5 most common image sizes:")
    common_sizes = Counter(cal_size_counter).most_common(5)
    for size, count in common_sizes:
        print(f"Size: {size}, Count: {count}")

    print("=" * 40)
