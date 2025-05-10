"""
This module contains custom dataset classes and utility functions.
"""
import os
import json
import random

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image


class TrainDataset(Dataset):
    def __init__(self, img_dir, json_path, transform=None):
        self.degraded_dir = os.path.join(img_dir, "degraded")
        self.clean_dir = os.path.join(img_dir, "clean")
        self.transform = transform

        # load the JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)

        # get the training images
        self.degraded_files = data

    def __len__(self):
        return len(self.degraded_files)

    def __getitem__(self, idx):
        degraded_filename = self.degraded_files[idx]
        degradation_type, image_id = degraded_filename.split("-", 1)
        clean_filename = f"{degradation_type}_clean-{image_id}"

        degraded_path = os.path.join(self.degraded_dir, degraded_filename)
        clean_path = os.path.join(self.clean_dir, clean_filename)

        degraded_img = Image.open(degraded_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        if self.transform:
            degraded_img = self.transform(degraded_img)
            clean_img = self.transform(clean_img)
            # Randomly crop the images
            degraded_img, clean_img = random_crop(
                degraded_img, clean_img, crop_size=256
            )

        return degraded_img, clean_img


class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.file_names = [f for f in os.listdir(img_dir) if f.endswith(".png")]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        degraded_filename = self.file_names[idx]
        degraded_path = os.path.join(self.img_dir, degraded_filename)
        degraded_img = Image.open(degraded_path).convert("RGB")

        if self.transform:
            degraded_img = self.transform(degraded_img)
        return degraded_img, degraded_filename
    
def create_folder_if_not_exists(folder_path):
    """
    Create a folder if it does not exist.
    Args:
        folder_path (str): The path to the folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Create a folder: {folder_path}")

def random_crop(x, label, crop_size):
    """x 和 label 是同一張圖的 degraded 和 ground truth，需同步 crop"""
    _, h, w = x.shape  # assume shape is [C, H, W]

    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)

    x_crop = x[:, top:top + crop_size, left:left + crop_size]
    label_crop = label[:, top:top + crop_size, left:left + crop_size]

    return x_crop, label_crop


def show_process(
    data,
    xname,
    yname,
    labels,
    file_name,
    folder_path,
    if_label=True,
    if_limit=False,
    y_lim=1,
):
    """
    Visualizes the input data and saves the plot as a PNG file.

    Args:
        data (numpy.ndarray): The data to plot. Should be a 2D array with shape.
        xname (str): Label for the x-axis.
        yname (str): Label for the y-axis.
        labels (list): List of labels for each line in the plot.
        file_name (str): The name of the file (without extension) to save the plot.
        folder_path (str): The path to the folder where the plot will be saved.
        if_label (bool, optional): Whether to display the legend on the plot. Default is True.
        if_limit (bool, optional): Whether to limit the y-axis range. Default is False.
        y_lim (float, optional): The upper limit for the y-axis when if_limit is True. Default is 1.
    """
    # Clear the current figure
    plt.clf()

    if if_limit:
        plt.ylim(0, y_lim)

    # Set the color palette
    colors = plt.get_cmap("Set1").colors

    for i in range(data.shape[0]):
        plt.plot(np.squeeze(data[i]), color=colors[i % len(colors)], label=labels[i])

    plt.xlabel(f"{xname}")
    plt.ylabel(f"{yname}")

    # if show labels or not
    if if_label:
        plt.legend()

    # save the figure
    create_folder_if_not_exists(folder_path)
    file_path = os.path.join(folder_path, f"{yname}_{file_name}.png")
    plt.savefig(file_path)
    print(f"Save the figure to {file_path}")
