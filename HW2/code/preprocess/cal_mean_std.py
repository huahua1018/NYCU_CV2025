"""
This script calculates the mean and standard deviation of RGB channels
for images in given dataset folders (train, valid, test). These statistics
are commonly used for normalizing image data before training deep learning models.
"""
import os

import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


def calculate_mean_std(dataset_path):
    """
    Calculates the per-channel mean and standard deviation for all images in a folder.

    Args:
        dataset_path (str): Path to the dataset folder containing images.

    Returns:
        mean (Tensor): Mean of R, G, B channels across all images.
        std (Tensor): Standard deviation of R, G, B channels across all images.
    """
    # Initialize tensors to store the accumulated mean and standard deviation
    sum_mean = torch.zeros(3)
    sum_std = torch.zeros(3)

    # Get all image file paths in the dataset directory
    image_paths = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path)]
    num_images = len(image_paths)

    # Define a transform to convert images to tensors
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts image to Tensor, pixel values range from [0, 1]
        ]
    )

    for image_path in tqdm(image_paths, desc="Calculating mean and std"):
        img = Image.open(image_path).convert("RGB")  # Convert image to RGB
        img_tensor = transform(img)  # Transform image to tensor in [0, 1] range

        # Accumulate mean and std for each channel (R, G, B)
        sum_mean += img_tensor.mean(
            dim=[1, 2]
        )  # Mean over height and width for each channel
        sum_std += img_tensor.std(dim=[1, 2])  # Std over height and width for each channel

    # Compute overall dataset mean and standard deviation
    sum_mean /= num_images
    sum_std /= num_images

    return sum_mean, sum_std


# Define dataset folders
folder = [
    "../../nycu-hw2-data/train/",
    "../../nycu-hw2-data/valid/",
    "../../nycu-hw2-data/test/",
]
for f in folder:
    print(f"folder: {f}")
    mean, std = calculate_mean_std(f)
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std}")
