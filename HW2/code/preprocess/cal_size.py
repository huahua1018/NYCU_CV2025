import os
from PIL import Image
from tqdm import tqdm
import numpy as np


def analyze_image_sizes(folder_path):
    widths, heights = [], []
    size_counter = {}
    ratio = []

    # 遍歷所有圖片
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    size_counter[(w, h)] = size_counter.get((w, h), 0) + 1
                    ratio.append(min(w, h) / max(w, h))
            except Exception as e:
                print(f"Error loading image {filename}: {e}")

    return widths, heights, size_counter, ratio

folder = ["../../nycu-hw2-data/train/", "../../nycu-hw2-data/valid/", "../../nycu-hw2-data/test/"]
for f in folder:
    print(f"folder: {f}")
    widths, heights, size_counter, ratio = analyze_image_sizes(f)
    print(f"圖像數量: {len(widths)}")
    print(f"最小尺寸: ({min(widths)}, {min(heights)})")
    print(f"最大尺寸: ({max(widths)}, {max(heights)})")
    print(f"平均尺寸: ({np.mean(widths):.2f}, {np.mean(heights):.2f})")
    print(f"最小比例: {min(ratio):.2f}")
    print(f"最大比例: {max(ratio):.2f}")
    print(f"平均比例: {np.mean(ratio):.2f}")

    # 常見前幾種尺寸
    from collections import Counter
    common_sizes = Counter(size_counter).most_common(5)
    print("最常見尺寸前五名：")
    for size, count in common_sizes:
        print(f"尺寸: {size}, 次數: {count}")
    print("=====================================")