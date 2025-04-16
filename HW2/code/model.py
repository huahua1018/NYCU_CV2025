"""
This script defines a custom Faster R-CNN model using a Swin Transformer backbone and 
implements several utility functions for training, evaluation, and testing.
"""

from typing import Tuple, List, Dict, Optional, Callable
from collections import OrderedDict
import json

from tqdm import tqdm
import pandas as pd
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2, FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import swin_t
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.poolers import MultiScaleRoIAlign
from torchvision.ops.giou_loss import generalized_box_iou_loss
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool, ExtraFPNBlock
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.roi_heads import RoIHeads
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

from utils import visualize_predictions


class SwinBackboneWithFPN(nn.Module):
    """
    Swin Transformer backbone with Feature Pyramid Network (FPN) for multi-scale feature extraction.

    Args:
        backbone: Swin Transformer backbone.
        return_layers: Layers to extract from the backbone.
        in_channels_list: Input channels for each selected layer.
        out_channels: Output channels for FPN layers.
        extra_blocks: Extra FPN operations (e.g., max pool).
        norm_layer (Optional[Callable]): Normalization layer for FPN.

    Returns:
        Dict[str, Tensor]: Multi-scale feature maps from FPN.
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
        """
        Initializes the Swin Transformer backbone with FPN.
        """
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
        """
        Performs the forward pass through the Swin backbone and FPN.
        """
        x = self.body(x)

        # From [B, H, W, C] to [B, C, H, W]
        x = {k: v.permute(0, 3, 1, 2) for k, v in x.items()}

        x = self.fpn(x)
        return x


def _swin_fpn_extractor(
    trainable_layers: int,
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[nn.Module] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
) -> nn.Module:
    """
    This function wraps a Swin-T backbone with an FPN, and allows selection of specific stages to be returned and trained.
    It also allows freezing of early layers for transfer learning, and optionally includes extra layers (e.g., max pooling) after the FPN.

    Args:
        trainable_layers: Number of Swin Transformer stages (from last to first) to set as trainable.
                            If 0, all layers are frozen.
        returned_layers: Indices (0-based) of the Swin stages to return as FPN inputs.
        extra_blocks: Additional layer(s) to apply after the FPN. Default is LastLevelMaxPool.
        norm_layer: Normalization layer to use in the FPN layers.
    """
    backbone = swin_t(weights="DEFAULT")
    backbone = backbone.features

    # These are the layers where Swin Transformer blocks are located.
    stage_indices = [1, 3, 5, 7]
    num_stages = len(stage_indices)

    # find the index of the layer from which we won't freeze
    if trainable_layers < 0 or trainable_layers > num_stages:
        raise ValueError(
            f"Trainable layers should be in the range [0,{num_stages}], got {trainable_layers} "
        )
    freeze_before = (
        len(backbone)
        if trainable_layers == 0
        else stage_indices[num_stages - trainable_layers]
    )

    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)

    out_channels = 256

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [num_stages - 2, num_stages - 1]
    if min(returned_layers) < 0 or max(returned_layers) >= num_stages:
        raise ValueError(
            f"Each returned layer should be in the range [0,{num_stages - 1}], got {returned_layers} "
        )

    return_layers = {
        f"{stage_indices[k]}": str(v) for v, k in enumerate(returned_layers)
    }

    # Number of channels corresponding to the layers mentioned in stage_indices
    in_channels_list = [96, 192, 384, 768]
    in_channels_list = [in_channels_list[i] for i in returned_layers]

    # Return the FPN-wrapped Swin Transformer backbone
    return SwinBackboneWithFPN(
        backbone,
        return_layers,
        in_channels_list,
        out_channels,
        extra_blocks=extra_blocks,
        norm_layer=norm_layer,
    )


class DecoupledFasterRCNNPredictor(nn.Module):
    """
    Custom predictor for Faster R-CNN
    Use different feature for classification and bounding box regression.
    Output:
        class scores and bounding box predictions
    """

    def __init__(self, in_channels, num_classes):
        """
        Args:
           in_channels (int): Number of input features.
           num_classes (int): Number of output classes (including background).
        """
        super().__init__()

        # Classification layer: outputs class scores
        self.cls_score = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, num_classes),
        )

        # Bounding box regression layer: outputs box coordinates
        self.bbox_pred = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, num_classes * 4),
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature tensor

        Returns:
            scores (Tensor): Class scores, shape (N, num_classes)
            bbox_deltas (Tensor): Bounding box predictions, shape (N, num_classes * 4)
        """
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        # Flatten input to (N, C)
        x = x.flatten(start_dim=1)

        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


class CustomFasterRCNNPredictor(nn.Module):
    """
    Custom predictor for Faster R-CNN
    Use same feature for classification and bounding box regression.

    Output:
        class scores and bounding box predictions
    """

    def __init__(self, in_channels, num_classes):
        """
        Args:
            in_channels (int): Number of input features.
            num_classes (int): Number of output classes (including background).
        """
        super().__init__()

        # Add a simple fully connected layer with ReLU and dropout
        self.linear = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Classification layer: outputs class scores
        self.cls_score = nn.Linear(in_channels, num_classes)

        # Bounding box regression layer: outputs box coordinates
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature tensor

        Returns:
            scores (Tensor): Class scores, shape (N, num_classes)
            bbox_deltas (Tensor): Bounding box predictions, shape (N, num_classes * 4)
        """
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting [1,1] but got {list(x.shape[2:])}",
            )

        # Flatten input to (N, C)
        x = x.flatten(start_dim=1)

        # Pass through linear layers
        x = self.linear(x)

        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


class CustomTwoMLPHead(nn.Module):
    """
    A custom two-layer MLP head used for object detection.
    Adds BatchNorm and Dropout for improved generalization.

    Args:
        in_channels (int): Number of input channels.
        representation_size (int): Output feature size for both MLP layers.
    """

    def __init__(self, in_channels, representation_size):
        """
        Initializes the MLP head.
        """
        super().__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.bn1 = nn.BatchNorm1d(representation_size)
        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(p=0.3)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.bn2 = nn.BatchNorm1d(representation_size)

    def forward(self, x):
        """
        Forward pass through the MLP head.
        """
        # Flatten input tensor
        x = x.flatten(start_dim=1)
        # FC -> BN -> ReLU -> Dropout
        x = F.relu(self.bn1(self.fc6(x)))
        x = self.dropout(x)
        # FC -> BN -> ReLU
        x = F.relu(self.bn2(self.fc7(x)))
        return x


class GIoURoIHeads(RoIHeads):
    """
    Replaces the default RoIHeads with a custom one that uses GIoU loss for bounding box regression.
    """

    def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
        """
        Computes the loss for Faster R-CNN with GIoU for bbox regression.
        """

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        classification_loss = F.cross_entropy(class_logits, labels)

        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        N, num_classes = class_logits.shape
        box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

        box_preds = box_regression[sampled_pos_inds_subset, labels_pos]

        # Use GIoU loss instead of smooth L1 loss
        box_loss = generalized_box_iou_loss(
            box_preds, regression_targets[sampled_pos_inds_subset], reduction="sum"
        )

        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss


def eval_forward(model, images, targets):
    """
    Performs a forward pass through the Faster R-CNN model for evaluation
    and got both losses and predictions.

    Reference:
    https://stackoverflow.com/questions/71288513/how-can-i-determine-validation-loss-for-faster-rcnn-pytorch
    """
    model.eval()

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    images, targets = model.transform(images, targets)

    # Check for degenerate boxes
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
    num_anchors_per_level = [
        s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors
    ]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(
        objectness, pred_bbox_deltas
    )
    # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # note that we detach the deltas because Faster R-CNN do not backprop through
    # the proposals
    proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    proposals, scores = model.rpn.filter_proposals(
        proposals, objectness, images.image_sizes, num_anchors_per_level
    )

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
    proposals_train, matched_idxs, labels, regression_targets = (
        model.roi_heads.select_training_samples(proposals, targets)
    )

    box_features = model.roi_heads.box_roi_pool(features, proposals_train, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)

    result: List[Dict[str, torch.Tensor]] = []
    detector_losses = {}
    loss_classifier, loss_box_reg = fastrcnn_loss(
        class_logits, box_regression, labels, regression_targets
    )
    detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
    # -----------------------------------

    box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)
    boxes, scores, labels = model.roi_heads.postprocess_detections(
        class_logits, box_regression, proposals, image_shapes
    )
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
    model.rpn.training = False
    model.roi_heads.training = False
    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)
    return losses, detections


class ModelFactory:
    """
    Responsible for returning the corresponding PyTorch model based on the name.
    """

    @staticmethod
    def get_model(model_name, num_classes=11, pretrained=True):
        """
        Returns the model based on the name.
        """
        if model_name == "fasterrcnn_swin_t_fpn":
            backbone = _swin_fpn_extractor(
                trainable_layers=3, norm_layer=nn.BatchNorm2d
            )

            # RPN setting
            anchor_sizes = (
                (
                    8,
                    16,
                    32,
                    64,
                    128,
                    256,
                    512,
                ),
            ) * 3
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

            # ROI setting
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
            )
            out_channels = backbone.out_channels
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = CustomTwoMLPHead(
                in_channels=out_channels * resolution**2,
                representation_size=representation_size,
            )
            box_predictor = DecoupledFasterRCNNPredictor(
                in_channels=representation_size, num_classes=num_classes
            )

            model = FasterRCNN(
                backbone,
                # RPN
                rpn_anchor_generator=anchor_generator,
                # ROI
                box_roi_pool=box_roi_pool,
                box_head=box_head,
                box_predictor=box_predictor,
            )

            model.roi_heads = GIoURoIHeads(
                box_roi_pool=model.roi_heads.box_roi_pool,
                box_head=model.roi_heads.box_head,
                box_predictor=model.roi_heads.box_predictor,
                fg_iou_thresh=0.5,
                bg_iou_thresh=0.5,
                batch_size_per_image=512,
                positive_fraction=0.25,
                bbox_reg_weights=None,
                score_thresh=0.05,
                nms_thresh=0.5,
                detections_per_img=100,
            )

        elif model_name in ["fasterrcnn_resnet50_fpn", "fasterrcnn_resnet50_fpn_v2"]:
            if model_name == "fasterrcnn_resnet50_fpn":
                base_model = fasterrcnn_resnet50_fpn(
                    weights="DEFAULT" if pretrained else None,
                    weights_backbone="DEFAULT" if pretrained else None,
                    trainable_backbone_layers=2,
                )

            elif model_name == "fasterrcnn_resnet50_fpn_v2":
                base_model = fasterrcnn_resnet50_fpn_v2(
                    weights="DEFAULT" if pretrained else None,
                    weights_backbone="DEFAULT" if pretrained else None,
                    trainable_backbone_layers=2,
                )

            model = FasterRCNN(
                backbone=base_model.backbone,
                num_classes=num_classes,
                rpn_anchor_generator=base_model.rpn.anchor_generator,
                box_roi_pool=base_model.roi_heads.box_roi_pool,
            )

            model.rpn = base_model.rpn
            model.roi_heads = base_model.roi_heads
            model.transform = base_model.transform

            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        else:
            raise ValueError(f"Model {model_name} is not supported.")

        return model


class ModelTrainer:
    """
    This class is responsible for training, evaluating and testing the model.
    """

    def __init__(
        self,
        model_name,
        num_classes=11,
        lr=1e-4,
        min_lr=1e-6,
        weight_decay=1e-4,
        factor=0.1,
        args=None,
    ):
        """
        Initializes the model trainer with the specified model name and number of classes.
        """
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ModelFactory.get_model(model_name, num_classes)
        self.model.to(self.device)

        self.args = args
        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.factor = factor
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
        obj_loss = 0
        rpn_loss = 0
        cls_loss = 0
        box_loss = 0
        pbar = tqdm(data_loader, ncols=120, desc=f"Epoch {epoch}")

        for img, img_id, target in pbar:
            img = [imgi.to(self.device) for imgi in img]
            target = [{k: v.to(self.device) for k, v in t.items()} for t in target]

            self.optim.zero_grad()

            loss_dict = self.model(img, target)

            # calculate loss
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

            obj_loss += loss_dict["loss_objectness"].item()
            rpn_loss += loss_dict["loss_rpn_box_reg"].item()
            cls_loss += loss_dict["loss_classifier"].item()
            box_loss += loss_dict["loss_box_reg"].item()

            losses.backward()
            self.optim.step()

            # get current learning rate and update tqdm
            lr = self.optim.param_groups[0]["lr"]
            pbar.set_postfix(loss=losses.item(), lr=lr)

        return (
            total_loss / len(data_loader),
            obj_loss / len(data_loader),
            rpn_loss / len(data_loader),
            cls_loss / len(data_loader),
            box_loss / len(data_loader),
            lr,
        )

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
        obj_loss = 0
        rpn_loss = 0
        cls_loss = 0
        box_loss = 0

        self.model.eval()
        pbar = tqdm(data_loader, ncols=120, desc=f"Epoch {epoch}", unit="batch")

        with torch.no_grad():
            for img, img_id, target in pbar:

                img = [imgi.to(self.device) for imgi in img]
                target = [{k: v.to(self.device) for k, v in t.items()} for t in target]

                loss_dict, output = eval_forward(self.model, img, target)

                # calculate loss
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()

                obj_loss += loss_dict["loss_objectness"].item()
                rpn_loss += loss_dict["loss_rpn_box_reg"].item()
                cls_loss += loss_dict["loss_classifier"].item()
                box_loss += loss_dict["loss_box_reg"].item()

                lr = self.optim.param_groups[0]["lr"]
                pbar.set_postfix(loss=losses.item(), lr=lr)

                # use for predict value
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
                            # add to calculate mAP
                            pred_list.append(
                                {
                                    "image_id": int(img_id[i]),
                                    "bbox": [x_min, y_min, width, height],
                                    "score": score,
                                    "category_id": digit,
                                }
                            )

                            detections.append({"x_min": x_min, "digit": digit})

                            # Use to draw on tensorboard
                            if draw:
                                draw_pred.append(
                                    {
                                        "bbox": [x_min, y_min, x_max, y_max],
                                        "category_id": digit - 1,
                                        "score": score,
                                    }
                                )

                    # Use to calculate value accuracy
                    if detections:
                        detections.sort(key=lambda d: d["x_min"])  # sort by x_min
                        pred_value = int(
                            "".join(str(d["digit"] - 1) for d in detections)
                        )
                    else:
                        pred_value = -1

                    pred_value_list.append([int(img_id[i]), pred_value])

                    # Draw on tensorboard
                    if draw:
                        visualize_predictions(
                            img[i], img_id[i], target[i], draw_pred, writer, epoch
                        )

        # Update scheduler
        self.scheduler.step(total_loss / len(data_loader))

        # Transform to DataFrame and save as json
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(pred_list, f, indent=4)

        # Transform to DataFrame and save as CSV
        df = pd.DataFrame(pred_value_list, columns=["image_id", "pred_label"])
        df.to_csv(csv_path, index=False)

        return (
            total_loss / len(data_loader),
            obj_loss / len(data_loader),
            rpn_loss / len(data_loader),
            cls_loss / len(data_loader),
            box_loss / len(data_loader),
        )

    def test(self, data_loader, json_path, csv_path, score_threshold, value_threshold):
        """
        Test the model on the test dataset and save the predictions to files.
        """
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
                    if score > score_threshold:
                        # add to calculate mAP
                        pred_list.append(
                            {
                                "image_id": int(img_id[i]),
                                "bbox": [x_min, y_min, width, height],
                                "score": score,
                                "category_id": digit,
                            }
                        )
                    if score > value_threshold:
                        detections.append({"x_min": x_min, "digit": digit})

                # Use to calculate value accuracy
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
        """
        Calculate the mean Average Precision (mAP) for the predictions on the validation set.
        Args:
            pred_file (str): Path to the predictions file.
            ground_truth_file (str): Path to the ground truth file.
        Returns:
            float: mAP score.
        """

        coco_gt = COCO(ground_truth_file)

        try:
            coco_dt = coco_gt.loadRes(pred_file)
        except IndexError:
            print("No valid annotations in predictions, skipping evaluation.")
            return 0.0

        # calculate mAP
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # AP @[ IoU=0.50:0.95 | area=all | maxDets=100 ]
        mAP = coco_eval.stats[0]

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
            optimizer, min_lr=self.min_lr, factor=self.factor, patience=2
        )

        return optimizer, scheduler
