"""
Split the dataset into train/val sets and convert the masks to COCO format.
"""

import os
import json
from pathlib import Path
from pycocotools.mask import area

import skimage.io as sio
import numpy as np
from sklearn.model_selection import train_test_split

from utils import encode_mask

# directory path
dir_root = "../hw3-data-release/"
data_root = Path(os.path.join(dir_root, "train"))
sample_dirs = [p for p in data_root.iterdir() if p.is_dir()]

split_dirs = [
    d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))
]
# 80% train / 20% valid
train_ids, val_ids = train_test_split(split_dirs, test_size=0.2, random_state=42)

# save for train/val
train_images = []
train_annotations = []
train_idx = 0
train_ins_idx = 0
val_images = []
val_annotations = []
val_idx = 0
val_ins_idx = 0
categories = [
    {
        "id": 1,
        "name": "class1",
    },
    {
        "id": 2,
        "name": "class2",
    },
    {
        "id": 3,
        "name": "class3",
    },
    {
        "id": 4,
        "name": "class4",
    },
]


for sample_dir in sample_dirs:

    # Only get dir name
    filename = sample_dir.name

    if filename in train_ids:
        print(f"Processing {filename} for training...")
        train_idx += 1
        train = True
        train_images.append(
            {
                "id": train_idx,
                "file_name": filename,
            }
        )
    else:
        print(f"Processing {filename} for validation...")
        val_idx += 1
        train = False
        val_images.append(
            {
                "id": val_idx,
                "file_name": filename,
            }
        )

    for class_id in range(1, 5):  # class1 ~ class4
        mask_path = sample_dir / f"class{class_id}.tif"
        if mask_path.exists():
            mask = sio.imread(str(mask_path))

            num_instance = np.max(mask).astype(np.int32)
            for inst_id in range(1, num_instance + 1):
                instance_mask = (mask == inst_id).astype(np.uint8)
                pos = np.where(instance_mask > 0)

                # Filter out empty masks
                if pos[0].size == 0 or pos[1].size == 0:
                    continue

                # Get the bounding box coordinates
                x_min = np.min(pos[1]).tolist()
                y_min = np.min(pos[0]).tolist()
                x_max = np.max(pos[1]).tolist()
                y_max = np.max(pos[0]).tolist()

                # Skip empty bounding boxes
                if x_max <= x_min or y_max <= y_min:
                    continue

                width = x_max - x_min
                height = y_max - y_min

                rle_mask = encode_mask(binary_mask=instance_mask)
                areas = area(rle_mask).tolist()

                if train:
                    train_ins_idx += 1
                    train_annotations.append(
                        {
                            "id": train_ins_idx,
                            "image_id": train_idx,
                            "bbox": [x_min, y_min, width, height],
                            "category_id": class_id,
                            "segmentation": rle_mask,
                            "area": areas,
                            "iscrowd": 0,
                        }
                    )
                else:
                    val_ins_idx += 1
                    val_annotations.append(
                        {
                            "id": val_ins_idx,
                            "image_id": val_idx,
                            "bbox": [x_min, y_min, width, height],
                            "category_id": class_id,
                            "segmentation": rle_mask,
                            "area": areas,
                            "iscrowd": 0,
                        }
                    )
train_content = {
    "images": train_images,
    "annotations": train_annotations,
    "categories": categories,
}
val_content = {
    "images": val_images,
    "annotations": val_annotations,
    "categories": categories,
}

# save json
train_json_path = os.path.join(dir_root, "train.json")
val_json_path = os.path.join(dir_root, "val.json")

# Transform to DataFrame and save as json
with open(train_json_path, "w", encoding="utf-8") as f:
    json.dump(train_content, f, indent=4)

with open(val_json_path, "w", encoding="utf-8") as f:
    json.dump(val_content, f, indent=4)
