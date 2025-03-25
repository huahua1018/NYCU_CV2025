"""
This module contains custom dataset classes and utility functions.

Classes:
    - TrainDataset: A custom dataset class for loading training images with labels.
    - TestDataset: A dataset class for loading test images without labels.

Functions:
    - create_folder_if_not_exists: Creates a folder at the specified path if it does not exist.
    - show_process: Plots and saves a figure of the given data.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class TrainDataset(Dataset):
    """
    A custom dataset class for loading training images.

    Attributes:
        data_dir (str): The directory where the image data is stored.
        transform (callable, optional): A function/transform to apply to the images.
        image_paths (list): A list of file paths to all images in the dataset.
        labels (list): A list of integer labels corresponding to the images.
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Traverse all the files in the image_dir
        for label in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label)
            for img_name in os.listdir(label_path):
                self.image_paths.append(os.path.join(label_path, img_name))
                self.labels.append(int(label))  # turn the label name into integer

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class TestDataset(Dataset):
    """
    A dataset class for loading test images without labels.

    Attributes:
        data_dir (str): The directory where the test images are stored.
        image_paths (list): A list of file names for the test images.
        transform (callable): A function/transform applied to the images before returning them.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = os.listdir(data_dir)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_paths[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, self.image_paths[idx]  # return img, img name


def create_folder_if_not_exists(folder_path):
    """
    Create a folder if it does not exist.
    Args:
        folder_path (str): The path to the folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Create a folder: {folder_path}")


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
