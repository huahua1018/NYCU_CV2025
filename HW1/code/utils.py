import os
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class TrainDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Traverse all the files in the image_dir
        for label in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label) 
            for img_name in os.listdir(label_path):
                self.image_paths.append(os.path.join(label_path, img_name))
                self.labels.append(int(label))  # turn the label name into integer

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
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = os.listdir(data_dir)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), # same as the validation set, otherwise the performance will be bad
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_paths[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, self.image_paths[idx]  # return img, img name
    

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
    