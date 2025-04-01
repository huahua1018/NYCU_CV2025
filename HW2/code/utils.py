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
import json
import cv2

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class TrainDataset(Dataset):
    """
    A custom dataset class for loading training images.
    """
    def __init__(self, json_path, img_dir, transform=None):
        """
        :param json_path: path to the JSON file containing image annotations
        :param img_dir: directory containing the images
        :param transform: transformations to be applied to the images
        """
        self.img_dir = img_dir
        self.transform = transform

        # read the JSON file
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # create a mapping from image id to file name
        self.img_id_to_name = {img["id"]: img["file_name"] for img in self.data["images"]}
        self.image_ids = list(self.img_id_to_name.keys())
        
        # create a mapping from image id to annotations,
        # ensuring that annotations for the same image are grouped together
        self.img_id_to_anns = {}
        for ann in self.data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_name = self.img_id_to_name[img_id]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        bboxes = []
        labels = []
        for ann in self.img_id_to_anns.get(img_id, []):
            bboxes.append(ann["bbox"])  # bbox : [x, y, w, h]
            labels.append(ann["category_id"])

        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        bboxes[:, 2] += bboxes[:, 0]  # x + w
        bboxes[:, 3] += bboxes[:, 1]  # y + h
        labels = torch.tensor(labels, dtype=torch.int64)

        # 建立 PyTorch Faster R-CNN 所需的 target 格式
        target = {
            "boxes": bboxes,   # (N, 4) 的 tensor
            "labels": labels   # (N,) 的 tensor
        }

        # 套用 transforms
        if self.transform:
            image = self.transform(image)

        # id to tensor
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
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = os.listdir(data_dir)
        self.transform = transforms.Compose(
            [
                # transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # ),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_paths[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        img_name = os.path.splitext(self.image_paths[idx])[0]
        return image, img_name  # return img, img name

def collate_fn(batch):
    images, id, targets = zip(*batch)  # 拆分圖像與標註    
    id = torch.stack(id, 0)  # 圖像 id shape 相同，可直接 stack

    return images, id, targets

def collate_fn_test(batch):
    images, id = zip(*batch)  # 拆分圖像與標註
    id = torch.stack(id, 0)  # 圖像 id shape 相同，可直接 stack
    return images, id

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


def visualize_predictions(image, id, targets, pred_list, writer, epoch):
    # 1. 原始圖像
    img_with_boxes = image.permute(1, 2, 0).cpu().numpy()  # [C, H, W] -> [H, W, C]
    img_with_boxes = np.ascontiguousarray(img_with_boxes)

    # 2. 預測結果 (這裡的 pred_list 是您提供的每個預測結果列表)
    for pred in pred_list:
        xmin, ymin, xmax, ymax = pred['bbox']
        
        # 畫出預測框
        cv2.rectangle(img_with_boxes, 
                      (int(xmin), int(ymin)), 
                      (int(xmax), int(ymax)), 
                      (255, 0, 0), 1)  # 用藍色畫框
        cv2.putText(img_with_boxes, 
                    f"{pred['category_id']}:{pred['score']:.2f}", 
                    (int(xmin), int(ymin)-2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.3, (255, 255, 255), 1)

    # 3. Ground truth 真實標註
    gt_boxes = targets['boxes']
    gt_labels = targets['labels']
    
    # 畫出真實框
    for box, label in zip(gt_boxes, gt_labels):
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(img_with_boxes, 
                      (int(xmin), int(ymin)), 
                      (int(xmax), int(ymax)), 
                      (0, 255, 0), 1)  # 用綠色畫框
        cv2.putText(img_with_boxes, 
                    f"{label.item()-1}",
                    (int(xmin), int(ymin)-2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.3, (0, 255, 0), 1)

    # 轉換回 [C, H, W] 並將圖像數據範圍從 [0, 255] -> [0.0, 1.0]
    img_with_boxes_tensor = torch.from_numpy(img_with_boxes).permute(2, 0, 1).float()

    # 4. 在 TensorBoard 上顯示圖片
    if writer is not None:
        writer.add_image(f'val{id}_pred', img_with_boxes_tensor, epoch)