"""
Main script for testing a trained model on a digit recognition task.

This script loads a trained model, 
runs inference on test data, and saves the results in both JSON and CSV formats.
Additionally, it zips the result files and stores them in a specified directory.
"""

import random
import argparse
import zipfile
import os

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

import utils
import model
from preprocess.clahe_and_sharpen import CLAHEandSharpen


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
    parser = argparse.ArgumentParser(description="Digit Recognition")

    parser.add_argument(
        "--test_data_path",
        type=str,
        default="../nycu-hw2-data/test/",
        help="Testing Data Path",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="../checkpoints/model_SwinTransformer_Combined_Decouple_weight_bs_8_epochs_18/mAP_best.pt",
        help="Path to checkpoint file.",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="../result/",
        help="Path to save the prediction result.",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker")
    parser.add_argument("--bs", type=int, default=1, help="Batch size for training.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="fasterrcnn_resnet50_fpn_v2",
        help="Model name to use for testing.",
    )
    # parser.add_argument(
    #     "--model_name",
    #     type=str,
    #     default="fasterrcnn_swin_t_fpn",
    #     help="Model name.",
    #     choices=[
    #         "fasterrcnn_resnet50_fpn",
    #         "fasterrcnn_resnet50_fpn_v2",
    #         "fasterrcnn_swin_t_fpn",
    #     ],
    # )
    parser.add_argument(
        "--resize",
        type=tuple,
        default=(200, 800),
        help="Resize the image to this size.",
    )
    parser.add_argument(
        "--num_classes", type=int, default=11, help="Number of classes."
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.0,
        help="Threshold for filtering detections.",
    )
    parser.add_argument(
        "--value_threshold",
        type=float,
        default=0.5,
        help="Threshold for filtering detections to produce value.",
    )

    args = parser.parse_args()

    # set seed
    setup_seed(118)

    model_name = os.path.basename(os.path.dirname(args.ckpt_path))
    save_dir = os.path.join(args.result_path, model_name)
    os.makedirs(save_dir, exist_ok=True)

    json_path = os.path.join(save_dir, "pred.json")
    csv_path = os.path.join(save_dir, "pred.csv")

    test_transform = transforms.Compose(
        [
            # CLAHEandSharpen(random_val=0),
            transforms.ToTensor(),
        ]
    )
    test_dataset = utils.TestDataset(
        data_dir=args.test_data_path, transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn_test,
    )

    myModel = model.ModelTrainer(
        model_name=args.model_name,
        num_classes=args.num_classes,
        args=args,
    )

    myModel.model.load_state_dict(torch.load(args.ckpt_path))
    myModel.model.to(args.device)
    myModel.test(
        test_loader, json_path, csv_path, args.score_threshold, args.value_threshold
    )

    # Create a zip file containing the JSON and CSV files
    zip_filename = os.path.join(save_dir, f"{model_name}.zip")
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_path, os.path.basename(json_path))
        zipf.write(csv_path, os.path.basename(csv_path))
