"""
This module contains custom dataset classes and utility functions.
"""
import os
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
import skimage.io as sio
from pycocotools import mask as mask_utils

def decode_maskobj(mask_obj):
    return mask_utils.decode(mask_obj)


def encode_mask(binary_mask):
    arr = np.asfortranarray(binary_mask).astype(np.uint8)
    rle = mask_utils.encode(arr)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def read_maskfile(filepath):
    mask_array = sio.imread(filepath)
    return mask_array

class TrainDataset(Dataset):
    def __init__(self, img_dir, json_path, transform=None):
        """
        Initialize the dataset.

        :param img_dir: Directory containing the images.
        :param json_path: Path to the JSON file containing image metadata.
        :param transform: Transformations to be applied to the images.
        """
        self.img_dir = img_dir
        self.transform = transform

        # Read the JSON file
        with open(json_path, "r") as f:
            self.data = json.load(f)

        # Create a mapping from image id to file name
        self.img_id_to_name = {
            img["id"]: img["file_name"] for img in self.data["images"]
        }
        self.image_ids = list(self.img_id_to_name.keys())

        # Create a mapping from image id to annotations,
        # ensuring that annotations for the same image are grouped together
        self.img_id_to_anns = {}
        for ann in self.data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)
    
    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """
        Get the image and its annotations at the given index.

        :param idx: Index of the image to retrieve.

        :return: A tuple (image, image_id, target), where:
            - image is the transformed image.
            - image_id is the ID of the image.
            - target is a dictionary containing 'boxes' and 'labels' for the image.
        """
        img_id = self.image_ids[idx]
        img_name = self.img_id_to_name[img_id]
        img_path = os.path.join(self.img_dir, img_name, "image.tif")

        image = Image.open(img_path).convert("RGB")

        bboxes = []
        labels = []
        masks = []
        for ann in self.img_id_to_anns.get(img_id, []):
            bboxes.append(ann["bbox"])  # bbox: [x, y, w, h]
            labels.append(ann["category_id"])
            mask = decode_maskobj(ann["segmentation"])
            masks.append(mask)

        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        bboxes[:, 2] += bboxes[:, 0]  # x + w
        bboxes[:, 3] += bboxes[:, 1]  # y + h
        labels = torch.tensor(labels, dtype=torch.int64)
        masks = np.array(masks)  # faster conversion
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        # Construct the target format required by PyTorch Faster R-CNN
        target = {
            "boxes": bboxes,  # Tensor of shape (N, 4)
            "labels": labels,  # Tensor of shape (N,)
            "masks": masks,  # List of masks
        }

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        # Convert image ID to tensor
        img_id = torch.tensor(img_id, dtype=torch.int64)
        return image, img_id, target


class TestDataset(Dataset):
    """
    A dataset class for loading test images without labels.

    Attributes:
        data_dir (str): The directory where the test images are stored.
        image_paths (list): A list of file names for the test images.
        transform (callable): A function/transform applied to the images before returning them.
    """

    def __init__(self, data_dir, json_path, transform=None):
        """
        Initializes the dataset.

        :param data_dir: Directory where the test images are stored.
        :param transform: Optional transformation to be applied to the images.
        """
        self.data_dir = data_dir

        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.transform = transform

    def __len__(self):
        """
        Return the total number of test images in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve the image at the given index.

        :param idx: Index of the image to retrieve.

        :return: A tuple (image, image_name), where:
            - image is the transformed test image.
            - image_name is the name of the image file (without extension).
        """
        sample = self.data[idx]
        file_name = sample["file_name"]
        image_id = sample["id"]

        img_path = os.path.join(self.data_dir, file_name)
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        return image, image_id  # Return the transformed image and image name


def create_folder_if_not_exists(folder_path):
    """
    Create a folder if it does not exist.
    Args:
        folder_path (str): The path to the folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Create a folder: {folder_path}")

def collate_fn(batch):
    """
    Custom collate function for training or validation.
    """
    # Unpack the batch into images, ids, and targets
    images, ids, targets = zip(*batch)

    # IDs can be stacked directly since they are of the same shape
    ids = torch.stack(ids, 0)

    return images, ids, targets

def collate_fn_test(batch):
    """
    Custom collate function for the test set.
    """
    # Unpack the batch into images and ids (no targets for test set)
    images, ids = zip(*batch)

    return images, ids

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
