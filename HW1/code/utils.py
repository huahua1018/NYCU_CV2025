import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np


from torch.utils.data import Dataset
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 遍歷資料夾，獲取所有圖片檔案及其標籤
        for label in os.listdir(image_dir):
            label_path = os.path.join(image_dir, label)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    self.image_paths.append(os.path.join(label_path, img_name))
                    self.labels.append(int(label))  # 假設標籤是資料夾名稱

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
class TestDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.image_paths = sorted(os.listdir(path))  # 確保讀取順序
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.image_paths[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, self.image_paths[idx]  # 回傳影像 + 檔名
    

# 創建資料夾的函式
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Create a folder: {folder_path}")

def show_process(data, xname, yname, labels, file_name, folder_path, if_label=True, if_limit=False, y_lim=1):
    plt.clf()   # Clear the current figure

    if if_limit:
        plt.ylim(0, y_lim)

    # 使用 matplotlib 的顏色方案
    colors = plt.get_cmap('Set1').colors 

    for i in range(data.shape[0]):
        plt.plot(np.squeeze(data[i]), color=colors[i % len(colors)], label=labels[i])

    plt.xlabel(f'{xname}')
    plt.ylabel(f"{yname}")
    if(if_label):
        plt.legend()    # show labels
    # plt.show()
    
    # save the figure
    create_folder_if_not_exists(folder_path)
    file_path = os.path.join(folder_path, f"{yname}_{file_name}.png")
    plt.savefig(file_path)
    print(f"Save the figure to {file_path}")
    