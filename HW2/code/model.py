import torch
from torch import optim, nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2, FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchvision.models import swin_t
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pandas as pd

from typing import Tuple, List, Dict, Optional, Callable
from collections import OrderedDict
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision.models.resnet import resnet50
from torchvision.models.mobilenetv2 import MobileNetV2, MobileNet_V2_Weights
from torchvision.models.swin_transformer import SwinTransformerBlock

from torch import Tensor
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool, ExtraFPNBlock
from torchvision.models._utils import IntermediateLayerGetter

import os
import json

from utils import visualize_predictions

from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

class SwinBackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediateLayerGetter apply here.
    Args:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(
        self,
        backbone: nn.Module,
        return_layers: Dict[str, str],
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.body(x)
        
        x = {k: v.permute(0, 3, 1, 2) for k, v in x.items()}

        x = self.fpn(x)
        return x

def _swin_extractor(
    trainable_layers: int,
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[nn.Module] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
) -> nn.Module:
    backbone = swin_t(weights="DEFAULT")
    backbone = backbone.features
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [1, 3, 5, 7]  # 這裡手動映射層
    num_stages = len(stage_indices)

    # find the index of the layer from which we won't freeze
    if trainable_layers < 0 or trainable_layers > num_stages:
        raise ValueError(f"Trainable layers should be in the range [0,{num_stages}], got {trainable_layers} ")
    freeze_before = len(backbone) if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)

    out_channels = 256

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [num_stages - 2, num_stages - 1]
    if min(returned_layers) < 0 or max(returned_layers) >= num_stages:
        raise ValueError(f"Each returned layer should be in the range [0,{num_stages - 1}], got {returned_layers} ")
    return_layers = {f"{stage_indices[k]}": str(v) for v, k in enumerate(returned_layers)}

    # print(f"Return layers: {return_layers}")
    # print(f"Returned layers: {returned_layers}")
    in_channels_list = [96, 192, 384, 768]  # Swin-T 這幾層的輸出通道數
    in_channels_list = [in_channels_list[i] for i in returned_layers]
    # print(f"In channels list: {in_channels_list}")

    return SwinBackboneWithFPN(
        backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer
    )
    
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
    
    # --------------------- only training ---------
    proposals_train, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(proposals, targets)

    box_features = model.roi_heads.box_roi_pool(features, proposals_train, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)

    result: List[Dict[str, torch.Tensor]] = []
    detector_losses = {}
    loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
    detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
    # -----------------------------------

    box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)
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

        self.model = FasterRCNN(
            backbone=base_model.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=base_model.rpn.anchor_generator,
            box_roi_pool=base_model.roi_heads.box_roi_pool,
            rpn_pre_nms_top_n_train=2000,
            rpn_pre_nms_top_n_test=2000,
            rpn_post_nms_top_n_train=2000,
            rpn_post_nms_top_n_test=2000,
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
        # # Swin Transformer 作為 backbone
        # backbone = _swin_extractor(trainable_layers=3)

        # anchor_sizes = (
        #     (
        #         8,
        #         16,
        #         32,
        #         64,
        #         128,
        #         256,
        #         512,
        #     ),
        # ) * 3
        
        # # anchor_sizes = ((8, 16,), (32, 64,), (128, 256))
        # aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

        # anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        

        # # # ROI Pooling 設定
        # # roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        # #     featmap_names=["0"],  # 只使用主要特徵圖
        # #     output_size=7,
        # #     sampling_ratio=2
        # # )

        # # 設定 Faster R-CNN
        # self.model = FasterRCNN(
        #     backbone,
        #     num_classes=num_classes,  # 數字 0~9 + 背景
        #     rpn_anchor_generator=anchor_generator,
        #     # box_roi_pool=roi_pooler
        #     rpn_pre_nms_top_n_train=2000,
        #     rpn_pre_nms_top_n_test=2000,
        #     rpn_post_nms_top_n_train=2000,
        #     rpn_post_nms_top_n_test=2000,
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
            # 假設 target 是包含多個張量的字典
            target = [{k: v.to("cuda:0") for k, v in t.items()} for t in target]

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

        self.model.eval()
        pbar = tqdm(data_loader, ncols=120, desc=f"Epoch {epoch}", unit="batch")

        with torch.no_grad():
            for img, img_id, target in pbar:

                img = [imgi.to(self.device) for imgi in img]
                # 假設 target 是包含多個張量的字典
                target = [{k: v.to(self.device) for k, v in t.items()} for t in target]

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

                        if score > self.args.score_threshold:
                            pred_list.append({
                                "image_id": int(img_id[i]),
                                "bbox": [x_min, y_min, width, height],
                                "score": score,
                                "category_id": digit
                            })

                            detections.append({"x_min": x_min, "digit": digit})

                            if draw:
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


        return total_loss / len(data_loader)

                
    def test(self, data_loader, json_path, csv_path):

        pred_list = []  # task 1
        pred_value_list = []  # task 2

        self.model.eval()
        pbar = tqdm(data_loader, ncols=120, desc="Predicting on data", unit="batch")

        for img, img_id in pbar:

            img = [imgi.to(self.device) for imgi in img]

            output = self.model(img)
            
            detections = []
            for i in range(len(img_id)):
                for j in range(len(output[i]["boxes"])):
                    x_min, y_min, x_max, y_max = output[i]["boxes"][j].tolist()
                    width = x_max - x_min
                    height = y_max - y_min
                    digit = output[i]["labels"][j].item()
                    score = output[i]["scores"][j].item()
                    if score > self.args.score_threshold:
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
        # coco_dt = coco_gt.loadRes(pred_file)

        try:
            coco_dt = coco_gt.loadRes(pred_file)
        except IndexError:
            print("No valid annotations in predictions, skipping evaluation.")
            return 0.0

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
