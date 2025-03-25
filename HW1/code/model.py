import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
import numpy as np

def replace_relu_with_leakyrelu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.LeakyReLU(negative_slope=0.01, inplace=True))
        else:
            replace_relu_with_leakyrelu(child)  # 遞迴替換
# ref: https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
def mixup_data(x, y, alpha=0.2):
    '''
    input: img, label, alpha
    mixed_x = lam * x1 + (1 - lam) * x2
    output: mixed inputs, pairs of targets, and lambda
    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]

    # x1 為照著原本順序的 x，此處打亂順序作為 x2
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

# ref: https://zhuanlan.zhihu.com/p/99261200
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(avg_out))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class Resnet_50(nn.Module):
    def __init__(self, args=None, num_classes=100, lr=1e-3, min_lr=1e-6, weight_decay=1e-4, factor=0.1):
        super().__init__()

        self.resnet = models.resnet50(weights=models.ResNet50Weights.DEFAULT)
        for name, param in self.resnet.named_parameters():
            if name.startswith("layer1") or name.startswith("layer2"):
                param.requires_grad = False  # 停止更新權重

        in_features = self.resnet.fc.in_features  # get the number of in_features of original fc layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5), 
            nn.Linear(in_features, num_classes)
        )
        
        # ResNet 的第一層
        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool

        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        # 加入 CBAM
        self.cbam1 = CBAM(in_planes=256)
        self.cbam2 = CBAM(in_planes=512)
        self.cbam3 = CBAM(in_planes=1024)
        self.cbam4 = CBAM(in_planes=2048)

        # doropout
        self.dropout = nn.Dropout(0.2)

        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.factor = factor

        self.loss_fn = nn.CrossEntropyLoss()
        self.args = args


        # 設置 optimizer 和 scheduler
        self.optim, self.scheduler = self.configure_optimizers()

    def forward(self, x):
        # 輸入第一層
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer1
        x = self.layer1(x)
        # x = self.cbam1(x)  # 加入 CBAM
        # x = self.dropout(x)

        # Layer2
        x = self.layer2(x)
        # x = self.cbam2(x)  # 加入 CBAM
        # x = self.dropout(x)

        # Layer3
        x = self.layer3(x)
        # x = self.cbam3(x)  # 加入 CBAM
        # x = self.dropout(x)


        # Layer4
        x = self.layer4(x)
        x = self.cbam4(x)  # 加入 CBAM
        # x = self.dropout(x)


        # 最後全連接層
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.96), weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=self.min_lr, factor=self.factor, patience=2)

        return optimizer, scheduler


class Resnext_50(nn.Module):
    def __init__(self, args=None, num_classes=100, lr=1e-3, min_lr=1e-6, weight_decay=1e-4, factor=0.1):
        super().__init__()

        self.resnext = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        # for name, param in self.resnext.named_parameters():
        #     if name.startswith("layer1") or name.startswith("layer2"):
        #         param.requires_grad = False  # 停止更新權重


        in_features = self.resnext.fc.in_features  # get the number of in_features of original fc layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5), 
            nn.Linear(in_features, num_classes)
        )
        
        # resnext 的第一層
        self.conv1 = self.resnext.conv1
        self.bn1 = self.resnext.bn1
        self.relu = self.resnext.relu
        self.maxpool = self.resnext.maxpool

        self.layer1 = self.resnext.layer1
        self.layer2 = self.resnext.layer2
        self.layer3 = self.resnext.layer3
        self.layer4 = self.resnext.layer4

        # 加入 CBAM
        self.cbam1 = CBAM(in_planes=256)
        self.cbam2 = CBAM(in_planes=512)
        self.cbam3 = CBAM(in_planes=1024)
        self.cbam4 = CBAM(in_planes=2048)

        # doropout
        self.dropout = nn.Dropout(0.2)

        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.factor = factor

        self.loss_fn = nn.CrossEntropyLoss()
        self.args = args


        # 設置 optimizer 和 scheduler
        self.optim, self.scheduler = self.configure_optimizers()

    def forward(self, x):
        # 輸入第一層
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer1
        x = self.layer1(x)
        # x = self.cbam1(x)  # 加入 CBAM
        # x = self.dropout(x)

        # Layer2
        x = self.layer2(x)
        # x = self.cbam2(x)  # 加入 CBAM
        # x = self.dropout(x)

        # Layer3
        x = self.layer3(x)
        # x = self.cbam3(x)  # 加入 CBAM
        # x = self.dropout(x)


        # Layer4
        x = self.layer4(x)
        # x = self.cbam4(x)  # 加入 CBAM
        # x = self.dropout(x)


        # 最後全連接層
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.96), weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=self.min_lr, factor=self.factor, patience=2)

        return optimizer, scheduler

class Resnext_101_32(nn.Module):
    def __init__(self, args=None, num_classes=100, lr=1e-3, min_lr=1e-6, weight_decay=1e-4, factor=0.1, mixup_alpha=0):
        super().__init__()

        self.resnext = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.DEFAULT)
        for name, param in self.resnext.named_parameters():
            if name.startswith("layer1") or name.startswith("layer2"):
                param.requires_grad = False  # 停止更新權重


        in_features = self.resnext.fc.in_features  # get the number of in_features of original fc layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5), 
            nn.Linear(in_features, num_classes)
        )
        
        # resnext 的第一層
        self.conv1 = self.resnext.conv1
        self.bn1 = self.resnext.bn1
        self.relu = self.resnext.relu
        self.maxpool = self.resnext.maxpool

        self.layer1 = self.resnext.layer1
        self.layer2 = self.resnext.layer2
        self.layer3 = self.resnext.layer3
        self.layer4 = self.resnext.layer4

        # 加入 CBAM
        self.cbam1 = CBAM(in_planes=256)
        self.cbam2 = CBAM(in_planes=512)
        self.cbam3 = CBAM(in_planes=1024)
        self.cbam4 = CBAM(in_planes=2048)

        # doropout
        self.dropout = nn.Dropout(0.2)

        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.factor = factor
        self.mixup_alpha = mixup_alpha

        self.loss_fn = nn.CrossEntropyLoss()
        self.args = args


        # 設置 optimizer 和 scheduler
        self.optim, self.scheduler = self.configure_optimizers()

    def forward(self, x):
        # 輸入第一層
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer1
        x = self.layer1(x)
        # x = self.cbam1(x)  # 加入 CBAM
        # x = self.dropout(x)

        # Layer2
        x = self.layer2(x)
        # x = self.cbam2(x)  # 加入 CBAM
        # x = self.dropout(x)

        # Layer3
        x = self.layer3(x)
        # x = self.cbam3(x)  # 加入 CBAM
        # x = self.dropout(x)


        # Layer4
        x = self.layer4(x)
        x = self.cbam4(x)  # 加入 CBAM


        # 最後全連接層
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    def train_one_epoch(self, data_loader, epoch):
        self.train()
        total_loss = 0
        pbar = tqdm(data_loader, ncols=120, desc=f"Epoch {epoch}")

        for img, label in pbar:
            img, label = img.to(self.args.device), label.to(self.args.device)
            self.optim.zero_grad()

            # random probability for mixup
            # mixup_pb = np.random.rand()
            mixup_pb = 1 # test without random
            if self.mixup_alpha != 0 and mixup_pb > 0.5:
                img, labels_a, labels_b, lam = mixup_data(img, label, alpha=self.mixup_alpha)
                pred = self.forward(img)
                # 計算 MixUp loss
                loss = lam * self.loss_fn(pred, labels_a) + (1 - lam) * self.loss_fn(pred, labels_b)
            else:
                pred = self.forward(img)
                loss = self.loss_fn(pred, label)

            # calculate loss and backpropagate
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.96), weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=self.min_lr, factor=self.factor, patience=3)

        return optimizer, scheduler


# 定義 SE Block
class SEBlock(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


# 封裝 Bottleneck Block，讓 ResNeXt-101 的 block 帶有 SE Block
class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, bottleneck_block):
        super(SEBottleneck, self).__init__()
        self.conv1 = bottleneck_block.conv1
        self.bn1 = bottleneck_block.bn1
        self.conv2 = bottleneck_block.conv2
        self.bn2 = bottleneck_block.bn2
        self.conv3 = bottleneck_block.conv3
        self.bn3 = bottleneck_block.bn3
        self.se = SEBlock(bottleneck_block.bn3.num_features)  # 加入 SE Block
        self.relu = bottleneck_block.relu
        self.downsample = bottleneck_block.downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)  # SE Block 在這裡作用於 block 內部

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# 修改 ResNeXt-101 架構，將 `layer1` ~ `layer4` 的 block 換成 `SEBottleneck`
class Resnext_101_32_SE(nn.Module):
    def __init__(self, args=None, num_classes=100, lr=1e-3, min_lr=1e-6, weight_decay=1e-4, factor=0.1):

        super().__init__()

        self.resnext = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.DEFAULT)

        in_features = self.resnext.fc.in_features  
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5), 
            nn.Linear(in_features, num_classes)
        )

        # 替換 ResNeXt-101 的 `layer1` ~ `layer4`，讓每個 block 都帶有 SE Block
        self.layer1 = self._replace_with_se(self.resnext.layer1)
        self.layer2 = self._replace_with_se(self.resnext.layer2)
        self.layer3 = self._replace_with_se(self.resnext.layer3)
        self.layer4 = self._replace_with_se(self.resnext.layer4)

        # 第一層保持不變
        self.conv1 = self.resnext.conv1
        self.bn1 = self.resnext.bn1
        self.relu = self.resnext.relu
        self.maxpool = self.resnext.maxpool

        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.factor = factor

        self.loss_fn = nn.CrossEntropyLoss()
        self.args = args


        # 設置 optimizer 和 scheduler
        self.optim, self.scheduler = self.configure_optimizers()

    def _replace_with_se(self, layer):
        """ 用 SEBottleneck 替換 ResNeXt layer 內的 block """
        se_blocks = []
        for bottleneck_block in layer:
            se_blocks.append(SEBottleneck(bottleneck_block))
        return nn.Sequential(*se_blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.96), weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=self.min_lr, factor=self.factor, patience=2)

        return optimizer, scheduler

