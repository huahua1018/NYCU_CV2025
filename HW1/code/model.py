import torch
import torch.nn as nn
from torchvision.models import resnet34
from tqdm import tqdm

class Resnet_34(nn.Module):
    def __init__(self, args=None, num_classes=100):
        super().__init__()

        self.model = resnet34(pretrained=True)
        in_features = self.model.fc.in_features  # 獲取 ResNet34 原始 fc 層的輸入維度 (512)
        self.model.fc = nn.Linear(in_features, num_classes)  # 替換原本的 fc 層
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.args = args

        # 設置 optimizer 和 scheduler
        self.optim, self.scheduler = self.configure_optimizers()

    def forward(self, x):
        return self.model(x)

    def train_one_epoch(self, data_loader, epoch):
        self.train()
        total_loss = 0
        pbar = tqdm(data_loader, ncols=120, desc=f"Epoch {epoch}")

        for img, label in pbar:
            img, label = img.to(self.args.device), label.to(self.args.device)

            self.optim.zero_grad()
            pred = self.forward(img)
            # 計算 loss
            loss = self.loss_fn(pred, label)
            loss.backward()
            self.optim.step()
            total_loss += loss.item()

            # 獲取當前學習率
            lr = self.optim.param_groups[0]['lr']
            
            # 更新 tqdm 進度條
            pbar.set_postfix(loss=loss.item(), lr=lr)

        # 根據 loss 更新 scheduler
        self.scheduler.step(total_loss / len(data_loader))
        return total_loss / len(data_loader)

    def eval_one_epoch(self, data_loader, epoch):
        self.eval()
        total_loss = 0
        total_acc = 0
        pbar = tqdm(data_loader, ncols=120, desc=f"Epoch {epoch}")

        with torch.no_grad():
            for img, label in pbar:
                img = img.to(self.args.device)
                label = label.to(self.args.device)

                pred = self.forward(img)
                loss = self.loss_fn(pred, label)
                total_loss += loss.item()

                acc = (pred.argmax(dim=1) == label).float().mean().item()
                total_acc += acc

                lr = self.optim.param_groups[0]['lr']
                pbar.set_postfix(loss=loss.item(), lr=lr, acc=acc)

        return total_loss / len(data_loader), total_acc / len(data_loader)
            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001, betas=(0.9, 0.96), weight_decay=4.5e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, factor=0.1, patience=2)
        return optimizer, scheduler
