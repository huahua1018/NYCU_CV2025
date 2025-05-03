"""
This script defines a custom Mask R-CNN model using a Swin Transformer backbone and 
implements several utility functions for training, evaluation, and testing.
"""

from typing import Tuple, List, Dict, Optional, Callable
from collections import OrderedDict
import json

from tqdm import tqdm
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2, MaskRCNN, MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FastRCNNConvFCHead
from torchvision.models import swin_t
from torchvision.models.detection.rpn import concat_box_prediction_layers, AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.poolers import MultiScaleRoIAlign
from torchvision.ops.giou_loss import generalized_box_iou_loss
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool, ExtraFPNBlock
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.roi_heads import RoIHeads, project_masks_on_boxes, fastrcnn_loss, maskrcnn_loss, maskrcnn_inference
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from torchvision.models.resnet import ResNet50_Weights

from utils import encode_mask


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

        self.cbam = CBAM(in_planes=out_channels)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Performs the forward pass through the Swin backbone and FPN.
        """
        x = self.body(x)

        # From [B, H, W, C] to [B, C, H, W]
        x = {k: v.permute(0, 3, 1, 2) for k, v in x.items()}

        x = self.fpn(x)

        for k, v in x.items():
            # Apply CBAM attention
            x[k] = self.cbam(v)

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


class CustomMaskRCNNHeads(nn.Sequential):
    """
    A custom Mask R-CNN head that stacks a series of convolutional layers
    followed by CBAM (Convolutional Block Attention Module) and dropout for regularization.
    """

    def __init__(self, in_channels, layers, dilation, norm_layer=None):
        blocks = []
        next_feature = in_channels
        for layer_features in layers:
            blocks.append(
                misc_nn_ops.Conv2dNormActivation(
                    next_feature,
                    layer_features,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    norm_layer=norm_layer,
                )
            )
            blocks.append(CBAM(layer_features))
            blocks.append(nn.Dropout2d(0.1))
        super().__init__(*blocks)


class CustomRoIHeads(RoIHeads):
    """
    Replaces the default RoIHeads with a custom one that uses original mask loss + dice loss for mask prediction.
    """

    def dice_loss(input, target, eps=1e-6):
        """
        Computes the Dice loss between the predicted and target masks.
        """
        input = torch.sigmoid(input)
        input = input.view(input.size(0), -1)
        target = target.view(target.size(0), -1)
        intersection = (input * target).sum(dim=1)
        union = input.sum(dim=1) + target.sum(dim=1)
        dice = (2 * intersection + eps) / (union + eps)
        return 1 - dice.mean()

    def maskrcnn_loss(
        self, mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs
    ):
        """
        Replaces the default maskrcnn_loss with a custom one that uses original mask loss + dice loss for mask prediction.
        """
        discretization_size = mask_logits.shape[-1]
        labels = [
            gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)
        ]
        mask_targets = [
            project_masks_on_boxes(m, p, i, discretization_size)
            for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
        ]

        labels = torch.cat(labels, dim=0)
        mask_targets = torch.cat(mask_targets, dim=0)

        if mask_targets.numel() == 0:
            return mask_logits.sum() * 0

        org_loss = F.binary_cross_entropy_with_logits(
            mask_logits[torch.arange(labels.shape[0], device=labels.device), labels],
            mask_targets,
        )
        # Make one-hot target for dice
        one_hot_targets = F.one_hot(labels, num_classes=5)  # [N, H, W, C]
        one_hot_targets = one_hot_targets.permute(0, 3, 1, 2).float()  # [N, C, H, W]

        dice = self.dice_loss(mask_logits, one_hot_targets)

        return org_loss + dice

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
    # -----------------------------------------------

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
    # -- mask --#
    mask_proposals_val = [p["boxes"] for p in result]

    # --------------------- only training ---------
    num_images = len(proposals)
    mask_proposals = []
    pos_matched_idxs = []
    for img_id in range(num_images):
        pos = torch.where(labels[img_id] > 0)[0]
        mask_proposals.append(proposals[img_id][pos])
        pos_matched_idxs.append(matched_idxs[img_id][pos])

    mask_features = model.roi_heads.mask_roi_pool(
        features, mask_proposals, image_shapes
    )
    mask_features = model.roi_heads.mask_head(mask_features)
    mask_logits = model.roi_heads.mask_predictor(mask_features)
    gt_masks = [t["masks"] for t in targets]
    gt_labels = [t["labels"] for t in targets]
    rcnn_loss_mask = maskrcnn_loss(
        mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs
    )
    loss_mask = {"loss_mask": rcnn_loss_mask}
    detector_losses.update(loss_mask)
    # ---------------------------------------------

    mask_features = model.roi_heads.mask_roi_pool(
        features, mask_proposals_val, image_shapes
    )
    mask_features = model.roi_heads.mask_head(mask_features)
    mask_logits = model.roi_heads.mask_predictor(mask_features)

    labels = [r["labels"] for r in result]
    masks_probs = maskrcnn_inference(mask_logits, labels)
    for mask_prob, r in zip(masks_probs, result):
        r["masks"] = mask_prob

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
    def get_model(model_name, num_classes=5, pretrained=True):
        """
        Returns the model based on the name.
        """
        if model_name == "maskrcnn_swin_t_fpn":
            backbone = _swin_fpn_extractor(
                trainable_layers=4, norm_layer=nn.BatchNorm2d
            )

            # RPN setting
            anchor_sizes = (
                (4, 8, 16, 32),
                (16, 32, 64, 128),
                (64, 128, 256, 512),
            )
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

            # ROI setting
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
            )
            out_channels = backbone.out_channels
            representation_size = 1024

            box_head = FastRCNNConvFCHead(
                (out_channels, 7, 7),
                [256, 256, 256, 256],
                [representation_size],
                norm_layer=nn.BatchNorm2d,
            )
            mask_head = CustomMaskRCNNHeads(
                out_channels, [256, 256, 256, 256], 1, norm_layer=nn.BatchNorm2d
            )

            box_predictor = FastRCNNPredictor(
                in_channels=representation_size, num_classes=num_classes
            )
            mask_predictor = MaskRCNNPredictor(256, 256, num_classes)

            model = MaskRCNN(
                backbone,
                # RPN
                rpn_anchor_generator=anchor_generator,
                # ROI
                box_roi_pool=box_roi_pool,
                box_head=box_head,
                box_predictor=box_predictor,
                # Mask
                mask_head=mask_head,
                mask_predictor=mask_predictor,
            )

            model.roi_heads = CustomRoIHeads(
                box_roi_pool=model.roi_heads.box_roi_pool,
                box_head=model.roi_heads.box_head,
                box_predictor=model.roi_heads.box_predictor,
                mask_roi_pool=model.roi_heads.mask_roi_pool,
                mask_head=model.roi_heads.mask_head,
                mask_predictor=model.roi_heads.mask_predictor,
                fg_iou_thresh=0.5,
                bg_iou_thresh=0.5,
                batch_size_per_image=512,
                positive_fraction=0.25,
                bbox_reg_weights=None,
                score_thresh=0.05,
                nms_thresh=0.5,
                detections_per_img=400,
            )

        elif model_name in ["maskrcnn_resnet50_fpn", "maskrcnn_resnet50_fpn_v2"]:
            if model_name == "maskrcnn_resnet50_fpn":
                base_model = maskrcnn_resnet50_fpn(
                    weights="DEFAULT" if pretrained else None,
                    weights_backbone=ResNet50_Weights.IMAGENET1K_V1
                    if pretrained
                    else None,
                    trainable_backbone_layers=3,
                )

            elif model_name == "maskrcnn_resnet50_fpn_v2":
                base_model = maskrcnn_resnet50_fpn_v2(
                    weights="DEFAULT" if pretrained else None,
                    weights_backbone=ResNet50_Weights.IMAGENET1K_V1
                    if pretrained
                    else None,
                    trainable_backbone_layers=3,
                )

            model = MaskRCNN(
                backbone=base_model.backbone,
                num_classes=num_classes,
            )
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
        epochs=100,
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
        self.epochs = epochs
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

        for img, _, target in pbar:
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
            mask_loss = loss_dict["loss_mask"].item()

            losses.backward()
            self.optim.step()

            # get current learning rate and update tqdm
            lr = self.optim.param_groups[0]["lr"]
            pbar.set_postfix(loss=losses.item(), lr=lr)

        # self.scheduler.step()

        return (
            total_loss / len(data_loader),
            obj_loss / len(data_loader),
            rpn_loss / len(data_loader),
            cls_loss / len(data_loader),
            box_loss / len(data_loader),
            mask_loss / len(data_loader),
            lr,
        )

    def eval_one_epoch(self, data_loader, epoch, json_path, writer):
        """
        Evaluates the model for one epoch.

        Args:
            data_loader (DataLoader): DataLoader for evaluation data.
            epoch (int): Current epoch number.

        """

        self.model.eval()
        pbar = tqdm(data_loader, ncols=120, desc=f"Epoch {epoch}", unit="batch")
        pred_list = []
        total_loss = 0
        obj_loss = 0
        rpn_loss = 0
        cls_loss = 0
        box_loss = 0
        mask_loss = 0

        with torch.no_grad():
            for img, img_id, target in pbar:

                img = [imgi.to(self.device) for imgi in img]
                target = [{k: v.to(self.device) for k, v in t.items()} for t in target]

                loss_dict, output = eval_forward(self.model, img, target)
                # output = self.model(img)

                # calculate loss
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()

                obj_loss += loss_dict["loss_objectness"].item()
                rpn_loss += loss_dict["loss_rpn_box_reg"].item()
                cls_loss += loss_dict["loss_classifier"].item()
                box_loss += loss_dict["loss_box_reg"].item()
                mask_loss = loss_dict["loss_mask"].item()

                for i in range(len(img)):

                    for j in range(len(output[i]["boxes"])):
                        x_min, y_min, x_max, y_max = output[i]["boxes"][j].tolist()
                        width = x_max - x_min
                        height = y_max - y_min
                        class_id = output[i]["labels"][j].item()
                        score = output[i]["scores"][j].item()
                        mask = output[i]["masks"][j].squeeze(0).detach().cpu().numpy()
                        binary_mask = mask > 0.5
                        rle_mask = encode_mask(binary_mask=binary_mask)
                        pred_list.append(
                            {
                                "image_id": int(img_id[i]),
                                "bbox": [x_min, y_min, width, height],
                                "score": score,
                                "category_id": class_id,
                                "segmentation": rle_mask,
                            }
                        )

        # Transform to DataFrame and save as json
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(pred_list, f, indent=4)

        mAP = self.calculate_mAP(
            pred_file=json_path,
            ground_truth_file=self.args.val_json_path,
        )
        # Update scheduler
        self.scheduler.step(mAP)

        return (
            total_loss / len(data_loader),
            obj_loss / len(data_loader),
            rpn_loss / len(data_loader),
            cls_loss / len(data_loader),
            box_loss / len(data_loader),
            mask_loss / len(data_loader),
            mAP,
        )

    def test(self, data_loader, json_path):
        """
        Test the model on the test dataset and save the predictions to files.
        """
        pred_list = []

        self.model.eval()
        pbar = tqdm(data_loader, ncols=120, desc="Predicting on data", unit="batch")

        for img, img_id in pbar:

            img = [imgi.to(self.device) for imgi in img]

            output = self.model(img)

            for i in range(len(img_id)):
                for j in range(len(output[i]["boxes"])):
                    x_min, y_min, x_max, y_max = output[i]["boxes"][j].tolist()
                    width = x_max - x_min
                    height = y_max - y_min
                    class_id = output[i]["labels"][j].item()
                    score = output[i]["scores"][j].item()
                    mask = output[i]["masks"][j].squeeze(0).detach().cpu().numpy()
                    binary_mask = mask > 0.5
                    rle_mask = encode_mask(binary_mask=binary_mask)
                    pred_list.append(
                        {
                            "image_id": int(img_id[i]),
                            "bbox": [x_min, y_min, width, height],
                            "score": score,
                            "category_id": class_id,
                            "segmentation": rle_mask,
                        }
                    )

        # Transform to DataFrame and save as json
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(pred_list, f, indent=4)

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
            optimizer,
            mode="max",  # mAP is a maximization metric
            min_lr=self.min_lr,
            factor=self.factor,
            patience=2,
        )

        return optimizer, scheduler
