import torch
import torch.nn as nn
from torchvision.models import resnet34, resnet50
from tqdm import tqdm

class Resnet_50(nn.Module):
    def __init__(self, args=None, num_classes=100, lr=1e-3, weight_decay=1e-4):
        super().__init__()

        self.model = resnet50(pretrained=True)
        in_features = self.model.fc.in_features  # get the number of in_features of original fc layer
        # self.model.fc = nn.Linear(in_features, num_classes)  # replace the fc layer
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5), 
            nn.Linear(in_features, num_classes)
        )
        
        self.lr = lr
        self.weight_decay = weight_decay

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

            # calculate loss and backpropagate
            loss = self.loss_fn(pred, label)
            loss.backward()
            self.optim.step()
            total_loss += loss.item()

            # get current learning rate and update tqdm
            lr = self.optim.param_groups[0]['lr']
            pbar.set_postfix(loss=loss.item(), lr=lr)

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

                # get prediction and calculate accuracy
                acc = (pred.argmax(dim=1) == label).float().mean().item()
                total_acc += acc

                lr = self.optim.param_groups[0]['lr']
                pbar.set_postfix(loss=loss.item(), lr=lr, acc=acc)

        # update scheduler
        self.scheduler.step(total_loss / len(data_loader))

        return total_loss / len(data_loader), total_acc / len(data_loader)
            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.96), weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, factor=0.8, patience=2)
        return optimizer, scheduler
