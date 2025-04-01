import torch
from torch import optim, nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2, FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
# from sklearn.metrics import average_precision_score
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pandas as pd

from typing import Tuple, List, Dict, Optional
from collections import OrderedDict
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.models.detection.rpn import AnchorGenerator

import os
import json

from utils import visualize_predictions

class CustomFasterRCNN(FasterRCNN):
    def eager_outputs(self, losses, detections):

        if self.training:
            return losses
        return detections


class ModelFactory:
    """ 
    Responsible for returning the corresponding PyTorch model based on the name.
    """
    
    @staticmethod
    def get_model(model_name, num_classes=11, pretrained=True):
        if model_name == "fasterrcnn_resnet50_fpn":
            model = fasterrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None, weights_backbone="DEFAULT" if pretrained else None, trainable_backbone_layers=2)

        elif model_name == "fasterrcnn_resnet50_fpn_v2":
            model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT" if pretrained else None, weights_backbone="DEFAULT" if pretrained else None, trainable_backbone_layers=2)
        
        else:
            raise ValueError(f"Model {model_name} is not supported.")
        
        return model

def eval_forward(model, images, targets):
    """
    Reference: https://stackoverflow.com/questions/71288513/how-can-i-determine-validation-loss-for-faster-rcnn-pytorch
    """
    model.eval()

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    images, targets = model.transform(images, targets)

    # Check for degenerate boxes
    # TODO: Move this to a function
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                raise ValueError(
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}."
                )

    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    model.rpn.training=True
    #model.roi_heads.training=True


    #####proposals, proposal_losses = model.rpn(images, features, targets)
    features_rpn = list(features.values())
    objectness, pred_bbox_deltas = model.rpn.head(features_rpn)
    anchors = model.rpn.anchor_generator(images, features_rpn)

    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # note that we detach the deltas because Faster R-CNN do not backprop through
    # the proposals
    proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    proposals, scores = model.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

    proposal_losses = {}
    assert targets is not None
    labels, matched_gt_boxes = model.rpn.assign_targets_to_anchors(anchors, targets)
    regression_targets = model.rpn.box_coder.encode(matched_gt_boxes, anchors)
    loss_objectness, loss_rpn_box_reg = model.rpn.compute_loss(
        objectness, pred_bbox_deltas, labels, regression_targets
    )
    proposal_losses = {
        "loss_objectness": loss_objectness,
        "loss_rpn_box_reg": loss_rpn_box_reg,
    }

    #####detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
    image_shapes = images.image_sizes
    proposals, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(proposals, targets)
    box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)

    result: List[Dict[str, torch.Tensor]] = []
    detector_losses = {}
    loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
    detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
    boxes, scores, labels = model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
    num_images = len(boxes)
    for i in range(num_images):
        result.append(
            {
                "boxes": boxes[i],
                "labels": labels[i],
                "scores": scores[i],
            }
        )
    detections = result
    detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]
    model.rpn.training=False
    model.roi_heads.training=False
    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)
    return losses, detections
    
class ModelTrainer:
    def __init__(
        self,
        model_name,
        num_classes=11,
        lr=1e-4,
        min_lr=1e-6,
        weight_decay=1e-4,
        factor=0.1,
        loss_lambda=10,
        args=None,
    ):
        """
        Initializes the model trainer with the specified model name and number of classes.
        """
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        #---------------------------- Pytorch ---------------------------------------#
        
        base_model = ModelFactory.get_model(model_name, num_classes)

        # 建立我們的自訂 Faster R-CNN
        self.model = CustomFasterRCNN(
            backbone=base_model.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=base_model.rpn.anchor_generator,
            box_roi_pool=base_model.roi_heads.box_roi_pool,
            box_score_thresh = 0.5 # 過濾掉低於 0.5 的預測
        )

        # 設定 model 的 RPN、ROI 頭部
        self.model.rpn = base_model.rpn
        self.model.roi_heads = base_model.roi_heads
        self.model.transform = base_model.transform

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model.to(self.device)
        #---------------------------- Pytorch ---------------------------------------#
        

        #---------------------------- Custom ---------------------------------------#
        # # 選擇 ResNet-50 並加上 FPN (提升小物件偵測能力)
        # backbone = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        # backbone = torch.nn.Sequential(*list(backbone.children())[:-2])  # 去掉全連接層
        # backbone.out_channels = 2048  # ResNet50 輸出通道數

        # # 設定小物件適合的 Anchor
        # anchor_generator = AnchorGenerator(
        #     sizes=((16, 32, 64, 128),),  # 適合數字的小型 anchor
        #     aspect_ratios=((0.5, 1.0, 2.0),)
        # )

        # # ROI Pooling 設定
        # roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        #     featmap_names=["0"],  # 只使用主要特徵圖
        #     output_size=7,
        #     sampling_ratio=2
        # )

        # # 設定 Faster R-CNN
        # self.model = FasterRCNN(
        #     backbone,
        #     num_classes=num_classes,  # 數字 0~9 + 背景
        #     rpn_anchor_generator=anchor_generator,
        #     box_roi_pool=roi_pooler
        # )
        # self.model.to(self.device)
        #---------------------------- Custom ---------------------------------------#


        self.args = args
        
        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.factor = factor
        self.loss_lambda = loss_lambda
        self.optim, self.scheduler = self.configure_optimizers()

    def train_one_epoch(self, data_loader, epoch):
        """
        Trains the model for one epoch.

        Args:
            data_loader (DataLoader): DataLoader for training data.
            epoch (int): Current epoch number.
        
        Returns:
            float: Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0
        pbar = tqdm(data_loader, ncols=120, desc=f"Epoch {epoch}")

        for img, img_id, target in pbar:
            # img = img.to(self.device)  # 將圖像移動到 GPU
            img = [imgi.to("cuda:0") for imgi in img]
            # target = [t.to("cuda:0") for t in target]
            # 假設 target 是包含多個張量的字典
            target = [{k: v.to("cuda:0") for k, v in t.items()} for t in target]


            # new_target = []
            # for t in target:
            #     # 获取 bounding boxes 和 labels
            #     bboxes = t['boxes']  # (N, 4) 的 tensor
            #     labels = t['labels']  # (N,) 的 tensor
                
            #     # 创建符合 Faster R-CNN 要求的格式
            #     target_dict = {
            #         "boxes": bboxes.to(self.device),  # (N, 4) 的 tensor
            #         "labels": labels.to(self.device),  # (N,) 的 tensor
            #     }
                
            #     # 将转换后的字典添加到 new_target 列表中
            #     new_target.append(target_dict)
            # target = new_target

            self.optim.zero_grad()  # 清除梯度

            # 前向傳播，傳遞圖像和標註
            loss_dict = self.model(img, target)  # 這裡傳入圖像和標註
            # 計算損失
            losses = loss_dict["loss_classifier"] + self.loss_lambda*loss_dict["loss_box_reg"]
            total_loss += losses.item()
            losses.backward()
            self.optim.step()

            # get current learning rate and update tqdm
            lr = self.optim.param_groups[0]["lr"]
            pbar.set_postfix(loss=losses.item(), lr=lr)


        return total_loss / len(data_loader), lr

    def eval_one_epoch(self, data_loader, epoch, json_path, csv_path, writer):
        """
        Evaluates the model for one epoch.

        Args:
            data_loader (DataLoader): DataLoader for evaluation data.
            epoch (int): Current epoch number.

        Returns:
            float: Average loss for the epoch.
            float: Average accuracy for the epoch.
        """

        pred_list = []  # task 1
        pred_value_list = []  # task 2
        total_loss = 0

        self.model.train()
        pbar = tqdm(data_loader, ncols=120, desc=f"Epoch {epoch}", unit="batch")

        with torch.no_grad():
            for img, img_id, target in pbar:

                # img = img.to(self.device)
                img = [imgi.to("cuda:0") for imgi in img]
                # target = [t.to("cuda:0") for t in target]
                # 假設 target 是包含多個張量的字典
                target = [{k: v.to("cuda:0") for k, v in t.items()} for t in target]
                # new_target = []
                # for t in target:
                #     # 获取 bounding boxes 和 labels
                #     bboxes = t['boxes']  # (N, 4) 的 tensor
                #     labels = t['labels']  # (N,) 的 tensor
                    
                #     # 创建符合 Faster R-CNN 要求的格式
                #     target_dict = {
                #         "boxes": bboxes.to(self.device),  # (N, 4) 的 tensor
                #         "labels": labels.to(self.device),  # (N,) 的 tensor
                #     }
                    
                #     # 将转换后的字典添加到 new_target 列表中
                #     new_target.append(target_dict)
                # target = new_target

                loss_dict, output = eval_forward(self.model, img, target)

                # 計算損失
                losses = loss_dict["loss_classifier"] + self.loss_lambda*loss_dict["loss_box_reg"]
                total_loss += losses.item()    

                lr = self.optim.param_groups[0]["lr"]
                pbar.set_postfix(loss=losses.item(), lr=lr)
                
                detections = []
                for i in range(len(img_id)):
                    if img_id[i] < 10:
                        draw = True
                        draw_pred = []
                    else:
                        draw = False

                    for j in range(len(output[i]["boxes"])):
                        x_min, y_min, x_max, y_max = output[i]["boxes"][j].tolist()
                        width = x_max - x_min
                        height = y_max - y_min
                        digit = output[i]["labels"][j].item()
                        score = output[i]["scores"][j].item()

                        pred_list.append({
                            "image_id": int(img_id[i]),
                            "bbox": [x_min, y_min, width, height],
                            "score": score,
                            "category_id": digit
                        })

                        detections.append({"x_min": x_min, "digit": digit})

                        if draw and score > 0.5:
                            draw_pred.append({
                                "bbox": [x_min, y_min, x_max, y_max],
                                "category_id": digit-1,
                                "score": score
                            })

                    if detections:
                        detections.sort(key=lambda d: d["x_min"])  # sort by x_min
                        pred_value = int("".join(str(d["digit"] - 1) for d in detections))
                    else:
                        pred_value = -1

                    pred_value_list.append([int(img_id[i]), pred_value])
                    
                    if(draw):
                        visualize_predictions(img[i], img_id[i], target[i], draw_pred, writer, epoch)

                    


        # update scheduler
        self.scheduler.step(total_loss / len(data_loader))

        # Transform to DataFrame and save as json
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(pred_list, f, indent=4)

        # Transform to DataFrame and save as CSV
        df = pd.DataFrame(pred_value_list, columns=["image_id", "pred_label"])
        df.to_csv(csv_path, index=False)


        # return total_loss / len(data_loader), total_acc / len(data_loader)
        return total_loss / len(data_loader)

                
    def test(self, data_loader, json_path, csv_path):

        pred_list = []  # task 1
        pred_value_list = []  # task 2

        self.model.eval()
        pbar = tqdm(data_loader, ncols=120, desc="Predicting on data", unit="batch")

        for img, img_id in pbar:

            # img = img.to(self.device)
            img = [imgi.to("cuda:0") for imgi in img]

            output = self.model(img)
            
            detections = []
            for i in range(len(img_id)):
                for j in range(len(output[i]["boxes"])):
                    x_min, y_min, x_max, y_max = output[i]["boxes"][j].tolist()
                    width = x_max - x_min
                    height = y_max - y_min
                    digit = output[i]["labels"][j].item()
                    score = output[i]["scores"][j].item()

                    pred_list.append({
                        "image_id": int(img_id[i]),
                        "bbox": [x_min, y_min, width, height],
                        "score": score,
                        "category_id": digit
                    })

                    detections.append({"x_min": x_min, "digit": digit})

                if detections:
                    detections.sort(key=lambda d: d["x_min"])  # sort by x_min
                    pred_value = int("".join(str(d["digit"] - 1) for d in detections))
                else:
                    pred_value = -1

                pred_value_list.append([int(img_id[i]), pred_value])

        


        # Transform to DataFrame and save as json
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(pred_list, f, indent=4)

        # Transform to DataFrame and save as CSV
        df = pd.DataFrame(pred_value_list, columns=["image_id", "pred_label"])
        df.to_csv(csv_path, index=False)

            
    def calculate_mAP(self, pred_file, ground_truth_file):
        # 建立 COCO 實例
        coco_gt = COCO(ground_truth_file)
        coco_dt = coco_gt.loadRes(pred_file)

        # 計算 mAP
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        mAP = coco_eval.stats[0]  # AP @[ IoU=0.50:0.95 | area=all | maxDets=100 ]

        return mAP



    def configure_optimizers(self):
        """
        Sets the optimizer and scheduler for the model.

        Returns:
            tuple: Optimizer and scheduler
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.96),
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            min_lr=self.min_lr,
            factor=self.factor,
            patience=2
        )

        return optimizer, scheduler
