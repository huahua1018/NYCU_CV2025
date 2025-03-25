"""
This module contains the implementation of the ResNet-50, ResNeXt-50, and ResNeXt-101 models
with CBAM (Convolutional Block Attention Module) for image classification.

Additionally, it includes a mixup data augmentation function
to improve model generalization and robustness.

The models include training and evaluation methods with learning rate scheduling.

"""

import torch
import numpy as np
from torchvision import models
from torch import nn
from tqdm import tqdm


def mixup_data(x, y, alpha=0.2):
    """
    ref: https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
    input: img, label, alpha
    mixed_x = lam * x1 + (1 - lam) * x2
    output: mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]

    # x1 is the original x, x2 is the shuffled x
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CA) enhances important feature channels using 
    both average-pooling and max-pooling, followed by a shared MLP.
    ref: https://zhuanlan.zhihu.com/p/99261200

    Args:
        in_planes (int): Number of input channels.
        ratio (int): Reduction ratio for the MLP. Default is 16.

    Forward Pass:
        x (Tensor): Input feature map of shape (B, C, H, W).
    
    Returns:
        Tensor: Attention-weighted feature map of the same shape as input.
    """
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the Channel Attention Module.
        """
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(avg_out))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SA) enhances important spatial regions 
    by computing attention using max-pooling and average-pooling along 
    the channel axis.
    ref: https://zhuanlan.zhihu.com/p/99261200

    Args:
        kernel_size (int): Kernel size for the convolutional layer. Default is 7.

    Forward Pass:
        x (Tensor): Input feature map of shape (B, C, H, W).
    
    Returns:
        Tensor: Spatial attention map of shape (B, 1, H, W).
    """
    def __init__(self, kernel_size=7):
        super().__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the Spatial Attention Module.
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) applies both channel and spatial attention 
    sequentially to refine feature representations.

    Args:
        in_planes (int): Number of input channels.
        ratio (int): Reduction ratio for channel attention. Default is 16.
        kernel_size (int): Kernel size for spatial attention. Default is 7.

    Forward Pass:
        x (Tensor): Input feature map of shape (B, C, H, W).
    
    Returns:
        Tensor: Attention-refined feature map of shape (B, C, H, W).
    """
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Forward pass of the Convolutional Block Attention Module.
        """
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class Resnet50(nn.Module):
    """
    Modified ResNet-50 with CBAM for image classification.

    Freezes `layer1` and `layer2`, adds CBAM after `layer4`, 
    and uses an FC layer for classification.
    Includes training and evaluation methods with LR scheduling.

    Args:
        args (Namespace, optional): Configuration parameters (e.g., device).
        num_classes (int, default=100): Number of output classes.
        lr (float, default=1e-4): Initial learning rate.
        min_lr (float, default=1e-6): Minimum learning rate.
        weight_decay (float, default=1e-4): Weight decay for optimizer.
        factor (float, default=0.1): LR reduction factor.
    """
    def __init__(
        self,
        args=None,
        num_classes=100,
        lr=1e-4,
        min_lr=1e-6,
        weight_decay=1e-4,
        factor=0.1,
    ):
        super().__init__()

        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Stop updating the weights of layer1 and layer2
        for name, param in self.resnet.named_parameters():
            if name.startswith("layer1") or name.startswith("layer2"):
                param.requires_grad = False

        in_features = (
            self.resnet.fc.in_features
        )  # get the number of in_features of original fc layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, num_classes))

        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool

        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        # CBAM
        self.cbam4 = CBAM(in_planes=2048)

        self.dropout = nn.Dropout(0.2)

        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.factor = factor

        self.loss_fn = nn.CrossEntropyLoss()
        self.args = args

        # Set optimizer and scheduler
        self.optim, self.scheduler = self.configure_optimizers()

    def forward(self, x):
        """"
        Forward pass of the ResNet-50 model."
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x = self.dropout(x)

        x = self.layer4(x)
        x = self.cbam4(x)

        # FC layer
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def train_one_epoch(self, data_loader, epoch):
        """
        Trains the model for one epoch.

        Args:
            data_loader (DataLoader): DataLoader for training data.
            epoch (int): Current epoch number.
        
        Returns:
            float: Average loss for the epoch.
        """
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
            lr = self.optim.param_groups[0]["lr"]
            pbar.set_postfix(loss=loss.item(), lr=lr)

        return total_loss / len(data_loader)

    def eval_one_epoch(self, data_loader, epoch):
        """
        Evaluates the model for one epoch.

        Args:
            data_loader (DataLoader): DataLoader for evaluation data.
            epoch (int): Current epoch number.

        Returns:
            float: Average loss for the epoch.
            float: Average accuracy for the epoch.
        """
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

                lr = self.optim.param_groups[0]["lr"]
                pbar.set_postfix(loss=loss.item(), lr=lr, acc=acc)

        # update scheduler
        self.scheduler.step(total_loss / len(data_loader))

        return total_loss / len(data_loader), total_acc / len(data_loader)

    def configure_optimizers(self):
        """
        Sets the optimizer and scheduler for the model.

        Returns:
            tuple: Optimizer and scheduler
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.96),
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, min_lr=self.min_lr, factor=self.factor, patience=2
        )

        return optimizer, scheduler


class Resnext50(nn.Module):
    """
    ResNeXt-50 model with CBAM attention mechanism and fine-tuned training process.

    Args:
        args (Namespace, optional): Arguments including training device.
        num_classes (int, optional): Number of output classes. Default is 100.
        lr (float, optional): Initial learning rate. Default is 1e-3.
        min_lr (float, optional): Minimum learning rate for scheduler. Default is 1e-6.
        weight_decay (float, optional): Weight decay for optimizer. Default is 1e-4.
        factor (float, optional): Factor for learning rate reduction. Default is 0.1.
    """
    def __init__(
        self,
        args=None,
        num_classes=100,
        lr=1e-3,
        min_lr=1e-6,
        weight_decay=1e-4,
        factor=0.1,
    ):
        super().__init__()

        self.resnext = models.resnext50_32x4d(
            weights=models.ResNeXt50_32X4D_Weights.DEFAULT
        )

        # Stop updating the weights of layer1 and layer2
        for name, param in self.resnext.named_parameters():
            if name.startswith("layer1") or name.startswith("layer2"):
                param.requires_grad = False

        in_features = (
            self.resnext.fc.in_features
        )  # get the number of in_features of original fc layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, num_classes))

        self.conv1 = self.resnext.conv1
        self.bn1 = self.resnext.bn1
        self.relu = self.resnext.relu
        self.maxpool = self.resnext.maxpool

        self.layer1 = self.resnext.layer1
        self.layer2 = self.resnext.layer2
        self.layer3 = self.resnext.layer3
        self.layer4 = self.resnext.layer4

        # CBAM
        self.cbam4 = CBAM(in_planes=2048)

        self.dropout = nn.Dropout(0.2)

        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.factor = factor

        self.loss_fn = nn.CrossEntropyLoss()
        self.args = args

        # Set optimizer and scheduler
        self.optim, self.scheduler = self.configure_optimizers()

    def forward(self, x):
        """
        Forward pass of the ResNeXt-50 model.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x = self.dropout(x)

        x = self.layer4(x)
        x = self.cbam4(x)

        # FC layer
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def train_one_epoch(self, data_loader, epoch):
        """
        Trains the model for one epoch.

        Args:
            data_loader (DataLoader): DataLoader for training data.
            epoch (int): Current epoch number.
        
        Returns:
            float: Average loss for the epoch.
        """
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
            lr = self.optim.param_groups[0]["lr"]
            pbar.set_postfix(loss=loss.item(), lr=lr)

        return total_loss / len(data_loader)

    def eval_one_epoch(self, data_loader, epoch):
        """
        Evaluates the model for one epoch.

        Args:
            data_loader (DataLoader): DataLoader for evaluation data.
            epoch (int): Current epoch number.

        Returns:
            float: Average loss for the epoch.
            float: Average accuracy for the epoch.
        """
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

                lr = self.optim.param_groups[0]["lr"]
                pbar.set_postfix(loss=loss.item(), lr=lr, acc=acc)

        # update scheduler
        self.scheduler.step(total_loss / len(data_loader))

        return total_loss / len(data_loader), total_acc / len(data_loader)

    def configure_optimizers(self):
        """
        Sets the optimizer and scheduler for the model.

        Returns:
            tuple: Optimizer and scheduler
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.96),
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, min_lr=self.min_lr, factor=self.factor, patience=2
        )

        return optimizer, scheduler


class Resnext101(nn.Module):
    """
    ResNeXt-101 model with CBAM attention and MixUp augmentation.
    
    This model is based on ResNeXt-101 (32x8d) and includes:
    - Feature extraction using ResNeXt-101 backbone
    - Channel and Spatial Attention (CBAM) in the last residual block
    - MixUp data augmentation during training

    Args:
        args (Namespace, optional): Configuration arguments including device information.
        num_classes (int, optional): Number of output classes. Default is 100.
        lr (float, optional): Initial learning rate. Default is 1e-3.
        min_lr (float, optional): Minimum learning rate for scheduling. Default is 1e-6.
        weight_decay (float, optional): Weight decay for optimizer. Default is 1e-4.
        factor (float, optional): Factor for ReduceLROnPlateau scheduler. Default is 0.1.
        mixup_alpha (float, optional): MixUp alpha value for data augmentation. Default is 0.
    """
    def __init__(
        self,
        args=None,
        num_classes=100,
        lr=1e-3,
        min_lr=1e-6,
        weight_decay=1e-4,
        factor=0.1,
        mixup_alpha=0,
    ):
        super().__init__()

        self.resnext = models.resnext101_32x8d(
            weights=models.ResNeXt101_32X8D_Weights.DEFAULT
        )

        # Stop updating the weights of layer1 and layer2
        for name, param in self.resnext.named_parameters():
            if name.startswith("layer1") or name.startswith("layer2"):
                param.requires_grad = False

        in_features = (
            self.resnext.fc.in_features
        )  # get the number of in_features of original fc layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, num_classes))

        self.conv1 = self.resnext.conv1
        self.bn1 = self.resnext.bn1
        self.relu = self.resnext.relu
        self.maxpool = self.resnext.maxpool

        self.layer1 = self.resnext.layer1
        self.layer2 = self.resnext.layer2
        self.layer3 = self.resnext.layer3
        self.layer4 = self.resnext.layer4

        # CBAM
        self.cbam4 = CBAM(in_planes=2048)

        self.dropout = nn.Dropout(0.2)

        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.factor = factor
        self.mixup_alpha = mixup_alpha

        self.loss_fn = nn.CrossEntropyLoss()
        self.args = args

        # Set optimizer and scheduler
        self.optim, self.scheduler = self.configure_optimizers()

    def forward(self, x):
        """
        Forward pass of the ResNeXt-101 model.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x = self.dropout(x)

        x = self.layer4(x)
        x = self.cbam4(x)  # use CBAM

        # FC layer
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def train_one_epoch(self, data_loader, epoch):
        """
        Trains the model for one epoch.

        Args:
            data_loader (DataLoader): DataLoader for training data.
            epoch (int): Current epoch number.
        
        Returns:
            float: Average loss for the epoch.
        """
        self.train()
        total_loss = 0
        pbar = tqdm(data_loader, ncols=120, desc=f"Epoch {epoch}")

        for img, label in pbar:
            img, label = img.to(self.args.device), label.to(self.args.device)
            self.optim.zero_grad()

            # random probability for mixup
            mixup_pb = np.random.rand()
            if self.mixup_alpha != 0 and mixup_pb > 0.5:
                img, labels_a, labels_b, lam = mixup_data(
                    img, label, alpha=self.mixup_alpha
                )
                pred = self.forward(img)
                # Calculate MixUp loss
                loss = lam * self.loss_fn(pred, labels_a) + (1 - lam) * self.loss_fn(
                    pred, labels_b
                )
            else:
                pred = self.forward(img)
                loss = self.loss_fn(pred, label)

            # calculate loss and backpropagate
            loss.backward()
            self.optim.step()
            total_loss += loss.item()

            # get current learning rate and update tqdm
            lr = self.optim.param_groups[0]["lr"]
            pbar.set_postfix(loss=loss.item(), lr=lr)

        return total_loss / len(data_loader)

    def eval_one_epoch(self, data_loader, epoch):
        """
        Evaluates the model for one epoch.

        Args:
            data_loader (DataLoader): DataLoader for evaluation data.
            epoch (int): Current epoch number.

        Returns:
            float: Average loss for the epoch.
            float: Average accuracy for the epoch.
        """
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

                lr = self.optim.param_groups[0]["lr"]
                pbar.set_postfix(loss=loss.item(), lr=lr, acc=acc)

        # update scheduler
        self.scheduler.step(total_loss / len(data_loader))

        return total_loss / len(data_loader), total_acc / len(data_loader)

    def configure_optimizers(self):
        """
        Sets the optimizer and scheduler for the model.

        Returns:
            tuple: Optimizer and scheduler
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.96),
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, min_lr=self.min_lr, factor=self.factor, patience=3
        )

        return optimizer, scheduler
