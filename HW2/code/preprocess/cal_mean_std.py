import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

def calculate_mean_std(dataset_path):
    # 初始化存儲所有圖像的均值和標準差
    mean = torch.zeros(3)
    std = torch.zeros(3)
    
    # 讀取所有圖像
    image_paths = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path)]
    num_images = len(image_paths)
    
    # 使用PyTorch的transform來將圖像轉換成Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),  # 轉換為Tensor，像素值範圍會是[0, 1]
    ])
    
    for image_path in tqdm(image_paths, desc="計算均值和標準差"):
        img = Image.open(image_path).convert('RGB')  # 轉換為RGB圖像
        img_tensor = transform(img)  # 將圖像轉換為Tensor，範圍是[0, 1]
        
        # 計算每個圖像每個通道的均值和標準差
        mean += img_tensor.mean(dim=[1, 2])  # 計算每個通道的均值
        std += img_tensor.std(dim=[1, 2])  # 計算每個通道的標準差
    
    # 計算整體均值和標準差
    mean /= num_images
    std /= num_images
    
    return mean, std

folder = ["../../nycu-hw2-data/train/", "../../nycu-hw2-data/valid/", "../../nycu-hw2-data/test/"]
for f in folder:
    print(f"folder: {f}")
    mean, std = calculate_mean_std(f)
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std}")
