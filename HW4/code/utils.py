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


class RandomCrop:
    """
    Randomly crop the input tensor and corresponding label tensor to a specified size.

    Args:
        crop_size (int): The size of the square crop (height and width will both be this size).
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, x, label):
        """
        Apply random crop to both image and label.

        Args:
            x (Tensor): Input image tensor with shape [C, H, W].
            label (Tensor): Corresponding label tensor with shape [C, H, W].

        Returns:
            Tuple[Tensor, Tensor]: Cropped image and label tensors.
        """
        _, h, w = x.shape  # Extract height and width

        # Randomly select the top-left corner of the crop
        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)

        # Perform the crop
        x_crop = x[:, top : top + self.crop_size, left : left + self.crop_size]
        label_crop = label[:, top : top + self.crop_size, left : left + self.crop_size]

        return x_crop, label_crop


class RandomFlip:
    """
    Randomly flip the input image and label horizontally with a given probability.

    Args:
        threshold (float): Probability threshold for applying the flip (default: 0.5).
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, x, label):
        """
        Apply random horizontal flip to both image and label.

        Args:
            x (Tensor): Input image tensor with shape [C, H, W].
            label (Tensor): Corresponding label tensor with shape [C, H, W].

        Returns:
            Tuple[Tensor, Tensor]: Possibly flipped image and label tensors.
        """
        if random.random() > self.threshold:
            # flip horizontally
            x = x.flip(2)
            label = label.flip(2)
        return x, label


class RandomRotate:
    """
    Randomly rotate the input image and label by a fixed angle with a given probability.

    Args:
        angle (float): The angle to rotate the image and label.
        threshold (float): Probability threshold for applying the rotation (default: 0.5).
    """

    def __init__(self, angle, threshold=0.5):
        self.angle = angle
        self.threshold = threshold

    def __call__(self, x, label):
        """
        Apply random rotation to both image and label.

        Args:
            x (PIL.Image or Tensor): Input image (must support `.rotate(angle)`).
            label (PIL.Image or Tensor): Corresponding label (must support `.rotate(angle)`).

        Returns:
            Tuple[Image, Image]: Possibly rotated image and label.
        """
        if random.random() > self.threshold:
            x = x.rotate(self.angle)  # Rotate image
            label = label.rotate(self.angle)
        return x, label


class TrainDataset(Dataset):
    """
    Custom dataset for loading pairs of degraded and clean images,
    with optional data augmentation.

    Args:
        transform (callable, optional): Transform to apply to both images.
        randomCrop (callable, optional): Function to apply random cropping.
        randomFlip (callable, optional): Function to apply random horizontal flipping.
        randomRotate (callable, optional): Function to apply random rotation.
    """

    def __init__(
        self,
        img_dir,
        json_path,
        transform=None,
        random_crop=None,
        random_flip=None,
        random_rotate=None,
    ):

        self.degraded_dir = os.path.join(img_dir, "degraded")
        self.clean_dir = os.path.join(img_dir, "clean")
        self.transform = transform
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.random_rotate = random_rotate

        # load the JSON file
        with open(json_path, "r") as f:
            data = json.load(f)

        # get the training images
        self.degraded_files = data

    def __len__(self):
        """Return the number of image pairs."""
        return len(self.degraded_files)

    def __getitem__(self, idx):
        """
        Load and return a pair of degraded and clean images, possibly with augmentations.

        Args:
            idx (int): Index of the image pair to load.

        Returns:
            Tuple[Tensor, Tensor]: A pair of transformed degraded and clean images.
        """
        # Get the degraded filename and parse it
        degraded_filename = self.degraded_files[idx]
        degradation_type, image_id = degraded_filename.split("-", 1)
        clean_filename = f"{degradation_type}_clean-{image_id}"

        degraded_path = os.path.join(self.degraded_dir, degraded_filename)
        clean_path = os.path.join(self.clean_dir, clean_filename)

        degraded_img = Image.open(degraded_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        if self.random_rotate:
            degraded_img, clean_img = self.random_rotate(
                degraded_img,
                clean_img,
            )
        if self.transform:
            degraded_img = self.transform(degraded_img)
            clean_img = self.transform(clean_img)
        if self.random_crop:
            degraded_img, clean_img = self.random_crop(
                degraded_img,
                clean_img,
            )
        if self.random_flip:
            degraded_img, clean_img = self.random_flip(
                degraded_img,
                clean_img,
            )

        return degraded_img, clean_img


class TestDataset(Dataset):
    """
    Custom dataset for loading test images from a directory.

    This dataset is used when only degraded images are available.

    Args:
        img_dir (str): Path to the directory containing test images (.png).
    """

    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.file_names = [f for f in os.listdir(img_dir) if f.endswith(".png")]

    def __len__(self):
        """Return the total number of images."""
        return len(self.file_names)

    def __getitem__(self, idx):
        """
        Load and return the test image at the specified index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            Tuple[Tensor, str]: The transformed image and its filename.
        """
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
