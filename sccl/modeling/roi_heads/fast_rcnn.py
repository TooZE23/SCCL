# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn

from detectron2.layers import cat, cross_entropy
from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputLayers,
    _log_classification_stats,
)

# focal loss
class FastRCNNFocaltLossOutputLayers(FastRCNNOutputLayers):
    def __init__(self, cfg, input_shape):
        super(FastRCNNFocaltLossOutputLayers, self).__init__(cfg, input_shape)
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        scores, proposal_deltas = predictions
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0)
            if len(proposals)
            else torch.empty(0, device=scores.device, dtype=torch.long)
        )
        _log_classification_stats(scores, gt_classes)

        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        losses = {
            "loss_cls": FocalLoss(gamma=1.5, num_classes=self.num_classes)(
                input=scores, target=gt_classes
            ),
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}


class FocalLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        gamma=1.0,
        num_classes=80,
    ):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

        self.num_classes = num_classes

    def forward(self, input, target):
        if target.numel() == 0:
            return input.sum() * 0.0
        # focal loss
        CE = cross_entropy(input, target, reduction="none")
        p = torch.exp(-CE)
        loss = (1 - p) ** self.gamma * CE
        return loss.mean()
