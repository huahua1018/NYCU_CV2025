import torch
import numpy as np
import random
import argparse

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import utils
import model

# Set seed for reproducibility
def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Classification")

    parser.add_argument('--train_path', type=str, default="../data/train/", help='Training Dataset Path')
    parser.add_argument('--val_path', type=str, default="../data/val/", help='Validation Dataset Path')
    parser.add_argument('--ckpt_path', type=str, default='../checkpoints/', help='Path to checkpoint folder.')
    parser.add_argument('--img_path', type=str, default='../img/', help='Path to save the image.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--bs', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0, help='Learning rate.')
    parser.add_argument('--num_classes', type=int, default=100, help='Number of classes.')
    parser.add_argument('--save_per_epoch', type=int, default=3, help='Save CKPT per ** epochs')
    parser.add_argument('--start_from_epoch', type=int, default=0, help='Begin training from this epoch.')
    args = parser.parse_args()

    # set seed
    setup_seed(118)

    # create checkpoint folder
    utils.create_folder_if_not_exists(args.ckpt_path)

    # 定義數據轉換
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 隨機裁剪 (保持 80% - 100% 的比例)
        transforms.RandomHorizontalFlip(p=0.5),  # 50% 機率水平翻轉
        transforms.RandomRotation(degrees=15),  # 隨機旋轉 ±15 度
        transforms.ToTensor(),          # 轉換為Tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 標準化
    ])

    # 建立Dataset
    # train_dataset = datasets.ImageFolder(root=args.train_path, transform=transfor
    train_dataset = utils.CustomImageDataset(image_dir=args.train_path, transform=transform)
    val_dataset = utils.CustomImageDataset(image_dir=args.val_path, transform=transform)

    # 檢查資料
    # print(f"Number of classes: {len(train_dataset.classes)}")
    # print(f"Class labels: {train_dataset.classes}")
    # print(f"First image path: {train_dataset.imgs[0][0]}, Label: {train_dataset.imgs[0][1]}")
    # for i in range(200):
    #     print(f"First image path: {train_dataset.imgs[i][0]}, Label: {train_dataset.imgs[i][1]}")
    
    
    # 建立DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)

    # 測試DataLoader
    # for i in range(2):
    #     for images, labels in train_loader:
    #         break


    # 建立模型
    mymodel = model.Resnet_34(args=args, num_classes=args.num_classes).to(args.device)
    print("Model parameters: ", sum(p.numel() for p in mymodel.parameters() if p.requires_grad))

    # training
    train_loss_proc = []
    val_loss_proc = []
    val_acc_proc = []
    highest_acc = 0

    mymodel.train()
    for epoch in range(args.start_from_epoch+1, args.epochs+1):

        train_loss=mymodel.train_one_epoch(train_loader,epoch)
        train_loss_proc.append(train_loss)

        val_loss, val_acc =mymodel.eval_one_epoch(val_loader,epoch)
        val_loss_proc.append(val_loss)
        val_acc_proc.append(val_acc)

        if epoch % args.save_per_epoch == 0:
            torch.save(mymodel.state_dict(), f"{args.ckpt_path}/epoch_{epoch}.pt")
        if val_acc > highest_acc:
            torch.save(mymodel.state_dict(), f"{args.ckpt_path}/best.pt")
            highest_acc = val_acc

        print(f"epoch {epoch}: train loss: {train_loss}, valid loss: {val_loss}, valid acc: {val_acc}")

    torch.save(mymodel.state_dict(), f"{args.ckpt_path}/epoch_{args.epochs}.pt")

    # Plot the training and validation loss
    utils.show_process(data=np.array([train_loss_proc]), xname="Epochs",yname="Loss", labels=np.array(["Loss"]), file_name="Training", folder_path=args.img_path,if_label=False)
    utils.show_process(data=np.array([val_loss_proc]), xname="Epochs",yname="Loss", labels=np.array(["Loss"]), file_name="Validation", folder_path=args.img_path,if_label=False)
    utils.show_process(data=np.array([val_acc_proc]), xname="Epochs",yname="Accuracy", labels=np.array(["Accuracy"]), file_name="Validation", folder_path=args.img_path,if_label=False)
    utils.show_process(data=np.array([train_loss_proc,val_loss_proc]), xname="Epochs",yname="Loss", labels=np.array(["Train","Valid"]), file_name="Training_Validation", folder_path=args.img_path,if_label=True)