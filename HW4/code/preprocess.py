"""
Split the dataset into train/val sets and convert the masks to COCO format.
"""

import os
import json
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit


# directory path
dir_root = "../hw4_realse_dataset/hw4_realse_dataset/"
data_root = Path(os.path.join(dir_root, "train/degraded"))
# get all sample filenames
filenames = [
    p.name for p in data_root.iterdir() if p.is_file() and p.name.endswith(".png")
]

# 'rain' or 'snow'
types = [f.split('-')[0] for f in filenames]  

# Stratified split 90% train / 10% valid
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train_idx, val_idx = next(splitter.split(filenames, types))

train_files = [filenames[i] for i in train_idx]
val_files = [filenames[i] for i in val_idx]


# save json
train_json_path = os.path.join(dir_root, "train.json")
val_json_path = os.path.join(dir_root, "val.json")

# Transform to DataFrame and save as json
with open(train_json_path, "w", encoding="utf-8") as f:
    json.dump(train_files, f, indent=4)

with open(val_json_path, "w", encoding="utf-8") as f:
    json.dump(val_files, f, indent=4)
