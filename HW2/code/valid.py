"""
This script is used to observe the performance of a trained model on validation data 
with different thresholds.
"""

import random
import argparse
import os

import torch
import numpy as np
from torch.utils.data import DataLoader

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
    parser = argparse.ArgumentParser(description="Digit Recognition")

    parser.add_argument(
        "--val_data_path",
        type=str,
        default="../nycu-hw2-data/valid/",
        help="Testing Data Path",
    )
    parser.add_argument(
        "--val_json_path",
        type=str,
        default="../nycu-hw2-data/valid.json",
        help="Validation JSON Path",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="../checkpoints/model/mAP_best.pt",
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
        default="fasterrcnn_resnet50_fpn",
        help="Model name to use for testing.",
    )
    parser.add_argument(
        "--num_classes", type=int, default=11, help="Number of classes."
    )

    args = parser.parse_args()

    # set seed
    setup_seed(118)

    model_name = os.path.basename(os.path.dirname(args.ckpt_path))
    save_dir = os.path.join(args.result_path, model_name, "val")
    os.makedirs(save_dir, exist_ok=True)

    val_dataset = utils.TestDataset(data_dir=args.val_data_path)
    val_loader = DataLoader(
        val_dataset,
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

    threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    max_mAP = 0
    best_t = 0

    for t in threshold:
        print(f"threshold: {t}")
        json_path = os.path.join(save_dir, f"pred_{t}.json")
        csv_path = os.path.join(save_dir, f"pred_{t}.csv")
        myModel.test(val_loader, json_path, csv_path, t, 0.5)

        mAP = myModel.calculate_mAP(
            pred_file=json_path,
            ground_truth_file=args.val_json_path,
        )
        print(f"mAP: {mAP}")
        if mAP > max_mAP:
            max_mAP = mAP
            best_t = t

    print(f"Best threshold: {best_t}: {max_mAP}")
