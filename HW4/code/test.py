"""
Main script for testing a trained model on a Image Restoration task.

This script loads a trained model, 
runs inference on test data, and saves the results in JSON formats.
Additionally, it zips the result files and stores them in a specified directory.
"""

import random
import argparse
import zipfile
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

import utils
import model


# Set seed for reproducibility
def setup_seed(seed):
    """
    Set the random seed for reproducibility across various libraries.

    This function ensures that the random number generators for PyTorch, NumPy, and Python's
    built-in `random` module are initialized with the given seed. It also configures CUDA-related
    settings to enhance reproducibility when using GPUs.

    Args:
        seed (int): The seed value to set for all random number generators.

    Returns:
        None
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="Image Restoration")

    parser.add_argument(
        "--test_data_path",
        type=str,
        default="../hw4_realse_dataset/hw4_realse_dataset/test/degraded/",
        help="Testing Data Path",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="../checkpoints/randcrop_bs_1_epochs_40_wd_0.005/PSNR_best.pt",
        help="Path to checkpoint file.",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="../result/",
        help="Path to save the prediction result.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Which device the training is on."
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker")
    parser.add_argument("--bs", type=int, default=1, help="Batch size for training.")

    args = parser.parse_args()

    # set seed
    setup_seed(118)

    model_name = os.path.basename(os.path.dirname(args.ckpt_path))
    save_dir = os.path.join(args.result_path, model_name)
    os.makedirs(save_dir, exist_ok=True)

    npz_path = os.path.join(save_dir, "pred.npz")

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    test_dataset = utils.TestDataset(
        img_dir=args.test_data_path,
        transform=test_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
    )

    myModel = model.ModelTrainer(
        args=args,
    )

    myModel.model.load_state_dict(torch.load(args.ckpt_path))
    myModel.model.to(args.device)
    myModel.test(test_loader, npz_path)

    # Create a zip file containing the npz file
    zip_filename = os.path.join(save_dir, f"{model_name}.zip")
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(npz_path, os.path.basename(npz_path))
