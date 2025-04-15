""" 
train.py: This script is responsible for training the model on the given dataset.

It loads data, applies transformations, initializes the model, trains it, 
and record performance.
"""

import os
import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import utils
import model
from preprocess.clahe_and_sharpen import CLAHEandSharpen


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
        "--train_data_path",
        type=str,
        default="../nycu-hw2-data/train/",
        help="Training Data Path",
    )
    parser.add_argument(
        "--train_json_path",
        type=str,
        default="../nycu-hw2-data/train.json",
        help="Training JSON Path",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default="../nycu-hw2-data/valid/",
        help="Validation Data Path",
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
        default="../checkpoints/",
        help="Path to checkpoint folder.",
    )
    parser.add_argument(
        "--img_path", type=str, default="../result/", help="Path to save the image."
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker")
    parser.add_argument(
        "--num_classes", type=int, default=11, help="Number of classes."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="fasterrcnn_swin_t_fpn",
        help="Model name.",
        choices=[
            "fasterrcnn_resnet50_fpn",
            "fasterrcnn_resnet50_fpn_v2",
            "fasterrcnn_swin_t_fpn",
        ],
    )
    parser.add_argument("--bs", type=int, default=8, help="Batch size for training.")
    parser.add_argument(
        "--epochs", type=int, default=18, help="Number of epochs to train."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--min_lr", type=float, default=1e-6, help="Minimum learning rate."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.005, help="Weight decay."
    )
    parser.add_argument(
        "--factor", type=float, default=0.1, help="Factor for ReduceLROnPlateau."
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.1,
        help="Score threshold for prediction."
    )

    parser.add_argument(
        "--save_per_epoch", type=int, default=1, help="Save CKPT per ** epochs"
    )
    parser.add_argument(
        "--start_from_epoch",
        type=int,
        default=0,
        help="Begin training from this epoch.",
    )
    args = parser.parse_args()

    # set seed
    setup_seed(118)

    # create log directory for tensorboard
    parm_dir = f"model_{args.model_name}_bs_{args.bs}_epochs_{args.epochs}"
    log_dir = os.path.join("runs", parm_dir)

    # create folder if not exists
    os.makedirs(log_dir, exist_ok=True)

    # create tensorboard writer
    writer = SummaryWriter(log_dir)

    # create checkpoint folder
    args.ckpt_path = os.path.join(args.ckpt_path, parm_dir)
    utils.create_folder_if_not_exists(args.ckpt_path)

    # create image folder
    args.img_path = os.path.join(args.img_path, parm_dir)
    utils.create_folder_if_not_exists(args.img_path)

    # define transform for training and validation
    # create model and compute the number of parameters
    myModel = model.ModelTrainer(
        model_name=args.model_name,
        num_classes=args.num_classes,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        factor=args.factor,
        args=args,
    )

    train_transform = transforms.Compose(
        [
            # CLAHEandSharpen(random_val=0.5),
            transforms.ToTensor(),
        ]
    )

    val_transform = transforms.Compose(
        [
            # CLAHEandSharpen(random_val=0),
            transforms.ToTensor(),
        ]
    )

    # create Dataset
    train_dataset = utils.TrainDataset(
        json_path=args.train_json_path,
        img_dir=args.train_data_path,
        transform=train_transform,
    )
    val_dataset = utils.TrainDataset(
        json_path=args.val_json_path,
        img_dir=args.val_data_path,
        transform=val_transform,
    )

    # create DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn,
    )

    with open(f"{args.img_path}/model_architecture.txt", "w", encoding="utf-8") as f:
        print(
            "Model parameters: ",
            sum(p.numel() for p in myModel.model.parameters() if p.requires_grad),
            file=f,
        )
        print(myModel.model, file=f)
    f.close()

    # begin training
    train_loss_proc = []
    val_loss_proc = []
    val_mAP_proc = []
    highest_mAP = 0
    lowest_loss = 1000000

    save_dir = os.path.join(args.img_path, "val")
    utils.create_folder_if_not_exists(save_dir)
    json_path = os.path.join(save_dir, "pred.json")
    csv_path = os.path.join(save_dir, "pred.csv")

    for epoch in range(args.start_from_epoch + 1, args.epochs + 1):

        # train_loss, train_lr = myModel.train_one_epoch(train_loader, epoch)
        (
            train_loss,
            train_obj_loss,
            train_rpn_loss,
            train_cls_loss,
            train_box_loss,
            train_lr,
        ) = myModel.train_one_epoch(train_loader, epoch)
        val_loss, val_obj_loss, val_rpn_loss, val_cls_loss, val_box_loss = (
            myModel.eval_one_epoch(val_loader, epoch, json_path, csv_path, writer)
        )
        mAP = myModel.calculate_mAP(
            pred_file=json_path,
            ground_truth_file=args.val_json_path,
        )

        # record the loss and accuracy
        train_loss_proc.append(train_loss)
        val_loss_proc.append(val_loss)
        val_mAP_proc.append(mAP)

        # write to tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/valid", val_loss, epoch)
        writer.add_scalar("Objectness Loss/train", train_obj_loss, epoch)
        writer.add_scalar("Objectness Loss/valid", val_obj_loss, epoch)
        writer.add_scalar("RPN Loss/train", train_rpn_loss, epoch)
        writer.add_scalar("RPN Loss/valid", val_rpn_loss, epoch)
        writer.add_scalar("Classification Loss/train", train_cls_loss, epoch)
        writer.add_scalar("Classification Loss/valid", val_cls_loss, epoch)
        writer.add_scalar("Box Loss/train", train_box_loss, epoch)
        writer.add_scalar("Box Loss/valid", val_box_loss, epoch)
        writer.add_scalar("mAP/valid", mAP, epoch)
        writer.add_scalar("Learning Rate", train_lr, epoch)

        writer.add_scalars("Loss", {"train": train_loss, "valid": val_loss}, epoch)
        writer.add_scalars(
            "Objectness Loss", {"train": train_obj_loss, "valid": val_obj_loss}, epoch
        )
        writer.add_scalars(
            "RPN Loss", {"train": train_rpn_loss, "valid": val_rpn_loss}, epoch
        )
        writer.add_scalars(
            "Classification Loss",
            {"train": train_cls_loss, "valid": val_cls_loss},
            epoch,
        )
        writer.add_scalars(
            "Box Loss", {"train": train_box_loss, "valid": val_box_loss}, epoch
        )

        # save the model
        if epoch % args.save_per_epoch == 0:
            torch.save(myModel.model.state_dict(), f"{args.ckpt_path}/epoch_{epoch}.pt")
        if mAP > highest_mAP:
            torch.save(myModel.model.state_dict(), f"{args.ckpt_path}/mAP_best.pt")
            highest_mAP = mAP
        if val_loss < lowest_loss:
            torch.save(myModel.model.state_dict(), f"{args.ckpt_path}/val_loss_best.pt")
            lowest_loss = val_loss

        print(
            f"epoch {epoch}: train loss: {train_loss}, valid loss: {val_loss}, valid mAP: {mAP}"
        )

    torch.save(myModel.model.state_dict(), f"{args.ckpt_path}/epoch_{args.epochs}.pt")

    # Plot the training and validation loss
    utils.show_process(
        data=np.array([train_loss_proc]),
        xname="Epochs",
        yname="Loss",
        labels=np.array(["Loss"]),
        file_name="Training",
        folder_path=args.img_path,
        if_label=False,
    )
    utils.show_process(
        data=np.array([val_loss_proc]),
        xname="Epochs",
        yname="Loss",
        labels=np.array(["Loss"]),
        file_name="Validation",
        folder_path=args.img_path,
        if_label=False,
    )
    utils.show_process(
        data=np.array([val_mAP_proc]),
        xname="Epochs",
        yname="mAP",
        labels=np.array(["mAP"]),
        file_name="Validation",
        folder_path=args.img_path,
        if_label=False,
    )
    utils.show_process(
        data=np.array([train_loss_proc, val_loss_proc]),
        xname="Epochs",
        yname="Loss",
        labels=np.array(["Train", "Valid"]),
        file_name="Training_Validation",
        folder_path=args.img_path,
        if_label=True,
    )

    # close tensorboard
    writer.close()
