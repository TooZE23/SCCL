# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from scipy.optimize import linear_sum_assignment
from detectron2.config import configurable
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
)
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from sccl_st.modeling.roi_heads.fast_rcnn import FastRCNNFocaltLossOutputLayers

import numpy as np
from detectron2.modeling.poolers import ROIPooler


@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsPseudoLab(StandardROIHeads):
    _OUTLIER_BUFFER_KEYS = (
        "target_outlier_buffer",
        "target_outlier_scores",
        "target_outlier_count",
    )

    @configurable
    def __init__(
        self,
        *,
        target_buffer_size: int = 32,
        source_center_momentum: float = 0.9,
        target_center_momentum: float = 0.9,
        contrastive_weight: float = 1.0,
        target_contrast_start_epoch: int = 1,
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 5,
        contrastive_eps: float = 1e-6,
        target_outlier_buffer_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        shape = self.box_head.output_shape
        feat_dim = int(shape.channels)
        if shape.height is not None and shape.width is not None:
            feat_dim *= int(shape.height) * int(shape.width)

        self.target_buffer_size = int(target_buffer_size)
        self.source_center_momentum = float(source_center_momentum)
        self.target_center_momentum = float(target_center_momentum)
        self.contrastive_weight = float(contrastive_weight)
        self.target_contrast_start_epoch = int(target_contrast_start_epoch)
        self.dbscan_eps = float(dbscan_eps)
        self.dbscan_min_samples = int(dbscan_min_samples)
        self.contrastive_eps = float(contrastive_eps)
        self.current_epoch = 0
        if target_outlier_buffer_size is None:
            target_outlier_buffer_size = int(target_buffer_size) * int(self.num_classes)
        self.target_outlier_buffer_size = int(target_outlier_buffer_size)
        self.target_init_started = False
        self.target_init_epoch = -1
        self.target_outlier_freed = False
        self._cached_source_features = None
        self._cached_source_labels = None

        self.register_buffer(
            "source_class_centers", torch.zeros(self.num_classes, feat_dim)
        )
        self.register_buffer(
            "source_class_counts", torch.zeros(self.num_classes)
        )
        self.register_buffer(
            "target_feature_buffer",
            torch.zeros(self.num_classes, self.target_buffer_size, feat_dim),
        )
        self.register_buffer(
            "target_feature_scores",
            torch.zeros(self.num_classes, self.target_buffer_size),
        )
        self.register_buffer(
            "target_buffer_ptr", torch.zeros(self.num_classes, dtype=torch.long)
        )
        self.register_buffer(
            "target_buffer_counts", torch.zeros(self.num_classes, dtype=torch.long)
        )
        self.register_buffer(
            "target_outlier_buffer",
            torch.zeros(self.target_outlier_buffer_size, feat_dim),
        )
        self.register_buffer(
            "target_outlier_scores",
            torch.zeros(self.target_outlier_buffer_size),
        )
        self.register_buffer(
            "target_outlier_count", torch.zeros(1, dtype=torch.long)
        )

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        roi_cfg = cfg.MODEL.ROI_HEADS
        target_contrast_start_epoch = int(cfg.TARGET_CONTRAST_START_EPOCH)
        ret.update(
            {
                "target_buffer_size": roi_cfg.get("TARGET_BUFFER_SIZE", 64),
                "source_center_momentum": roi_cfg.get("SOURCE_CENTER_MOMENTUM", 0.9),
                "target_center_momentum": roi_cfg.get("TARGET_CENTER_MOMENTUM", 0.9),
                "contrastive_weight": roi_cfg.get("CONTRASTIVE_WEIGHT", 1.0),
                "target_contrast_start_epoch": target_contrast_start_epoch,
                "dbscan_eps": roi_cfg.get("DBSCAN_EPS", 0.5),
                "dbscan_min_samples": roi_cfg.get("DBSCAN_MIN_SAMPLES", 5),
                "contrastive_eps": roi_cfg.get("CONTRASTIVE_EPS", 1e-6),
                "target_outlier_buffer_size": roi_cfg.get(
                    "TARGET_OUTLIER_BUFFER_SIZE", None
                ),
            }
        )
        return ret

    def set_current_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        for key in list(state.keys()):
            if any(key.endswith(name) for name in self._OUTLIER_BUFFER_KEYS):
                del state[key]
        return state

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for name in self._OUTLIER_BUFFER_KEYS:
            full_key = prefix + name
            if full_key in state_dict:
                del state_dict[full_key]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        missing_keys[:] = [
            key for key in missing_keys if not key.endswith(self._OUTLIER_BUFFER_KEYS)
        ]

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )
        if cfg.MODEL.ROI_HEADS.LOSS == "CrossEntropy":
            box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        elif cfg.MODEL.ROI_HEADS.LOSS == "FocalLoss":
            box_predictor = FastRCNNFocaltLossOutputLayers(cfg, box_head.output_shape)
        else:
            raise ValueError("Unknown ROI head loss.")

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        branch: str = "source",
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:

        del images
        if self.training:
            if branch == "target" and targets is None:
                pass
            else:
                assert targets, "'targets' argument is required during training"
                proposals = self.label_and_sample_proposals(
                    proposals, targets, branch=branch
                )
                # gt_boxes = [x.gt_boxes for x in targets]
                # gt_classes = torch.cat([x.gt_classes for x in targets], dim=0)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals, branch=branch)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals, branch=branch)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            return pred_instances, {}
        
    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances], branch: str = ""):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        box_features_flat = self._flatten_box_features(box_features)
        if self.training and branch in ("source", "source_contrast_only"):
            box_gt_cls = torch.cat([x.gt_classes for x in proposals], dim=0)
            
        if self.training:
            if branch in ("source", "source_contrast_only"):
                self._update_source_centers(box_features_flat, box_gt_cls)
                self._cached_source_features = box_features_flat
                self._cached_source_labels = box_gt_cls
            elif branch == "target":
                target_scores = self._get_target_objectness_scores(proposals)
                if self.current_epoch == self.target_contrast_start_epoch:
                    self._init_target_buffer_epoch(box_features_flat, target_scores)
                elif self.current_epoch > self.target_contrast_start_epoch:
                    if not self.target_outlier_freed:
                        self._free_target_outlier_buffers()
                    # 1) Assign pseudo labels with pre-update centers.
                    pseudo_labels = self._get_target_pseudo_labels(box_features_flat)
                    self._update_target_buffer(
                        box_features_flat, pseudo_labels, target_scores
                    )
                    # 2) Re-assign with updated centers for loss computation.
                    pseudo_labels = self._get_target_pseudo_labels(box_features_flat)

        del box_features

        if self.training:
            if branch == "target":
                losses = {}
                if self.current_epoch > self.target_contrast_start_epoch:
                    loss_contrast = self._target_cluster_contrast_loss(
                        box_features_flat, pseudo_labels
                    )
                    if loss_contrast is not None:
                        losses["loss_roi_contrast"] = (
                            loss_contrast * self.contrastive_weight
                        )
                    loss_center = self._cross_domain_center_mse(
                        box_features_flat, pseudo_labels
                    )
                    if loss_center is not None:
                        losses["loss_center_contrast"] = loss_center
                return losses

            if branch == "source_contrast_only":
                # Skip detection losses; only compute source contrast loss
                losses = {}
            else:
                losses = self.box_predictor.losses(predictions, proposals)

            if branch not in ("sourceonly",):
                loss_source_contrast = self._source_center_contrast_loss(
                    box_features_flat, box_gt_cls
                )
                if loss_source_contrast is not None:
                    losses["loss_source_contrast"] = (
                        loss_source_contrast * self.contrastive_weight
                    )
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes and branch != "source_contrast_only":
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    def _flatten_box_features(self, box_features: torch.Tensor) -> torch.Tensor:
        if box_features.dim() <= 2:
            return box_features
        return box_features.flatten(start_dim=1)

    def _get_target_objectness_scores(
        self, proposals: List[Instances]
    ) -> torch.Tensor:
        scores = [p.objectness_logits for p in proposals]
        scores = torch.cat(scores, dim=0)
        return scores.sigmoid()

    def _reset_target_init_state_if_needed(self) -> bool:
        if not self.target_init_started or self.target_init_epoch != self.current_epoch:
            self.target_init_started = True
            self.target_init_epoch = self.current_epoch
            self.target_outlier_count.zero_()
            self.target_buffer_counts.zero_()
            self.target_buffer_ptr.zero_()
            return True
        return False

    def _free_target_outlier_buffers(self) -> None:
        feat_dim = self.target_outlier_buffer.shape[1]
        device = self.target_outlier_buffer.device
        self.target_outlier_buffer = torch.empty((0, feat_dim), device=device)
        self.target_outlier_scores = torch.empty((0,), device=device)
        self.target_outlier_count.zero_()
        self.target_outlier_freed = True

    @torch.no_grad()
    def _flatten_target_buffer_with_scores(
        self,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        feats = []
        scores = []
        for cls_id in range(self.num_classes):
            count = int(self.target_buffer_counts[cls_id].item())
            if count == 0:
                continue
            feats.append(self.target_feature_buffer[cls_id, :count])
            scores.append(self.target_feature_scores[cls_id, :count])
        if not feats:
            return None, None
        return torch.cat(feats, dim=0), torch.cat(scores, dim=0)

    @torch.no_grad()
    def _init_target_buffer_epoch(
        self, features: torch.Tensor, scores: torch.Tensor
    ) -> None:
        is_first_iter = self._reset_target_init_state_if_needed()
        if features.numel() == 0:
            return

        feats = features.detach()
        scores = scores.detach()
        outlier_count = int(self.target_outlier_count.item())
        outlier_feats = None
        outlier_scores = None
        if outlier_count > 0:
            outlier_feats = self.target_outlier_buffer[:outlier_count]
            outlier_scores = self.target_outlier_scores[:outlier_count]

        if is_first_iter:
            cluster_feats = feats
            cluster_scores = scores
        else:
            mem_feats, mem_scores = self._flatten_target_buffer_with_scores()
            parts = [feats]
            score_parts = [scores]
            if mem_feats is not None:
                parts.append(mem_feats)
                score_parts.append(mem_scores)
            if outlier_feats is not None:
                parts.append(outlier_feats)
                score_parts.append(outlier_scores)
            cluster_feats = torch.cat(parts, dim=0)
            cluster_scores = torch.cat(score_parts, dim=0)

        if cluster_feats.shape[0] < self.dbscan_min_samples:
            return

        cluster_feats_norm = F.normalize(cluster_feats, dim=1)
        labels, num_clusters = self._dbscan(
            cluster_feats_norm, self.dbscan_eps, self.dbscan_min_samples
        )
        if num_clusters == 0:
            return

        # Reset buffers for this init iteration.
        self.target_buffer_counts.zero_()
        self.target_buffer_ptr.zero_()

        reserved_mask = torch.zeros_like(labels, dtype=torch.bool)
        cluster_centers = []
        cluster_quality = []
        for cluster_id in range(num_clusters):
            mask = labels == cluster_id
            if not mask.any():
                cluster_centers.append(None)
                cluster_quality.append(0.0)
                continue
            cluster_centers.append(cluster_feats_norm[mask].mean(dim=0))
            cluster_quality.append(float(mask.sum().item()) * float(cluster_scores[mask].mean().item()))

        sorted_ids = sorted(
            range(num_clusters), key=lambda k: cluster_quality[k], reverse=True
        )
        major_ids = sorted_ids[: min(self.num_classes, len(sorted_ids))]
        major_centers = [cluster_centers[k] for k in major_ids if cluster_centers[k] is not None]
        if major_centers:
            major_centers = torch.stack(major_centers, dim=0)
        else:
            major_centers = None
        major_map = {cid: i for i, cid in enumerate(major_ids)}

        candidates = [[] for _ in range(self.num_classes)]
        for i in range(labels.numel()):
            cid = int(labels[i].item())
            if cid < 0:
                continue
            if cid in major_map:
                candidates[major_map[cid]].append(i)
                continue
            if major_centers is None:
                continue
            dist = torch.cdist(
                cluster_feats_norm[i].unsqueeze(0), major_centers, p=2
            ).squeeze(0)
            min_dist, min_idx = dist.min(dim=0)
            if float(min_dist.item()) <= self.dbscan_eps:
                candidates[int(min_idx.item())].append(i)

        for class_idx in range(self.num_classes):
            if not candidates[class_idx]:
                continue
            idx = torch.tensor(candidates[class_idx], device=cluster_scores.device)
            cls_scores = cluster_scores[idx]
            order = torch.argsort(cls_scores, descending=True)
            keep = order[: self.target_buffer_size]
            keep_idx = idx[keep]

            count = keep_idx.numel()
            # Keep initialization scale consistent with later normalized updates.
            self.target_feature_buffer[class_idx, :count] = cluster_feats_norm[keep_idx]
            self.target_feature_scores[class_idx, :count] = cluster_scores[keep_idx]
            self.target_buffer_counts[class_idx] = count
            reserved_mask[keep_idx] = True

        # Update outliers: unclustered + unreserved clustered samples.
        outlier_mask = (labels == -1) | (~reserved_mask)
        outlier_idx = torch.nonzero(outlier_mask, as_tuple=False).squeeze(1)
        if outlier_idx.numel() == 0:
            self.target_outlier_count.zero_()
            return

        out_feats = cluster_feats_norm[outlier_idx]
        out_scores = cluster_scores[outlier_idx]
        order = torch.argsort(out_scores, descending=True)
        keep_num = max(1, int(order.numel() * 0.5))
        keep_idx = order[:keep_num]
        out_feats = out_feats[keep_idx]
        out_scores = out_scores[keep_idx]
        if out_feats.shape[0] > self.target_outlier_buffer_size:
            out_feats = out_feats[: self.target_outlier_buffer_size]
            out_scores = out_scores[: self.target_outlier_buffer_size]

        self.target_outlier_buffer[: out_feats.shape[0]] = out_feats
        self.target_outlier_scores[: out_scores.shape[0]] = out_scores
        self.target_outlier_count[0] = out_feats.shape[0]

    @torch.no_grad()
    def _update_source_centers(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        valid = (labels >= 0) & (labels < self.num_classes)
        if not valid.any():
            return

        # Keep centers on the same scale as contrastive distance computation.
        features = F.normalize(features[valid], dim=1)
        labels = labels[valid]
        for cls_id in labels.unique():
            cls_id = int(cls_id.item())
            cls_mask = labels == cls_id
            cls_size = int(cls_mask.sum().item())
            cls_center = features[cls_mask].mean(dim=0, keepdim=True)
            cls_center = F.normalize(cls_center, dim=1).squeeze(0)
            if self.source_class_counts[cls_id] == 0:
                self.source_class_centers[cls_id] = cls_center
            else:
                # Larger per-class batches should have proportionally larger impact.
                momentum = float(self.source_center_momentum) ** max(cls_size, 1)
                updated_center = (
                    self.source_class_centers[cls_id] * momentum
                    + cls_center * (1.0 - momentum)
                )
                self.source_class_centers[cls_id] = F.normalize(
                    updated_center.unsqueeze(0), dim=1
                ).squeeze(0)
            self.source_class_counts[cls_id] += cls_size

    @torch.no_grad()
    def _update_target_buffer(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> None:
        if scores is not None and scores.numel() != labels.numel():
            raise ValueError("scores and labels must have the same length")

        valid = (labels >= 0) & (labels < self.num_classes)
        if scores is not None:
            valid = valid & torch.isfinite(scores)
        if not valid.any():
            return

        # Normalize to align with distance-based pseudo labeling.
        features = F.normalize(features[valid], dim=1)
        labels = labels[valid]
        if scores is not None:
            scores = scores[valid]
        for cls_id in labels.unique():
            cls_id = int(cls_id.item())
            cls_mask = labels == cls_id
            cls_feats = features[cls_mask]
            cls_scores = scores[cls_mask] if scores is not None else None

            if cls_scores is not None:
                order = torch.argsort(cls_scores, descending=True)
                cls_feats = cls_feats[order]
                cls_scores = cls_scores[order]

            # Prevent a noisy large class from dominating one iteration update.
            max_updates = min(cls_feats.shape[0], int(self.target_buffer_size/2))
            cls_feats = cls_feats[:max_updates]
            if cls_scores is not None:
                cls_scores = cls_scores[:max_updates]

            count = int(self.target_buffer_counts[cls_id].item())
            ptr = int(self.target_buffer_ptr[cls_id].item())
            for i in range(cls_feats.shape[0]):
                idx = count if count < self.target_buffer_size else ptr
                score_val = (
                    float(cls_scores[i].item()) if cls_scores is not None else 1.0
                )
                if count < self.target_buffer_size:
                    self.target_feature_buffer[cls_id, idx] = cls_feats[i]
                    self.target_feature_scores[cls_id, idx] = score_val
                else:
                    self.target_feature_buffer[cls_id, idx] = (
                        self.target_feature_buffer[cls_id, idx]
                        * self.target_center_momentum
                        + cls_feats[i] * (1.0 - self.target_center_momentum)
                    )
                    self.target_feature_scores[cls_id, idx] = (
                        self.target_feature_scores[cls_id, idx]
                        * self.target_center_momentum
                        + score_val * (1.0 - self.target_center_momentum)
                    )

                if count < self.target_buffer_size:
                    count += 1
                else:
                    ptr = (ptr + 1) % self.target_buffer_size

            self.target_buffer_counts[cls_id] = count
            self.target_buffer_ptr[cls_id] = ptr

    def _cross_domain_center_mse(
        self, target_features: torch.Tensor, pseudo_labels: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Straight-through cross-domain center MSE.

        Forward value  = updated buffer centers (matches paper pseudocode).
        Backward grad  = flows through current-batch features via STE trick:
            c_st = c_batch + (c_buffer - c_batch).detach()
        so forward(c_st) == c_buffer, but grad(c_st) w.r.t. c_batch == I.
        """
        src_feats = self._cached_source_features
        src_labels = self._cached_source_labels
        self._cached_source_features = None
        self._cached_source_labels = None
        if src_feats is None or src_labels is None:
            return None

        valid_src = (src_labels >= 0) & (src_labels < self.num_classes)
        if not valid_src.any():
            return None
        src_feats = src_feats[valid_src]
        src_labels = src_labels[valid_src]

        valid_tgt = (pseudo_labels >= 0) & (pseudo_labels < self.num_classes)
        if not valid_tgt.any():
            return None
        tgt_feats = target_features[valid_tgt]
        tgt_labels = pseudo_labels[valid_tgt]

        # --- source centers (STE: forward=buffer, backward=batch) ---
        s_centers = []
        for cls_id in range(self.num_classes):
            has_buffer = self.source_class_counts[cls_id] > 0
            if not has_buffer:
                continue
            src_buffer_center = self.source_class_centers[cls_id]

            batch_mask = src_labels == cls_id
            if batch_mask.any():
                batch_center = src_feats[batch_mask].mean(dim=0)
                center = batch_center + (
                    src_buffer_center - batch_center
                ).detach()
            else:
                center = src_buffer_center.detach()
            s_centers.append(center)

        if not s_centers:
            return None
        source_centers = torch.stack(s_centers, dim=0)

        # --- target centers (STE: forward=buffer, backward=batch) ---
        # Target buffer slot IDs are DBSCAN-assigned, NOT aligned with
        # source GT class IDs, so we collect all target centers independently
        # and use one-to-one matching below.
        t_centers = []
        for slot_id in range(self.num_classes):
            buf_count = int(self.target_buffer_counts[slot_id].item())
            if buf_count == 0:
                continue
            tgt_buffer_center = self.target_feature_buffer[
                slot_id, :buf_count
            ].mean(dim=0)

            batch_mask = tgt_labels == slot_id
            if batch_mask.any():
                batch_center = tgt_feats[batch_mask].mean(dim=0)
                center = batch_center + (
                    tgt_buffer_center - batch_center
                ).detach()
            else:
                center = tgt_buffer_center.detach()
            t_centers.append(center)

        if not t_centers:
            return None
        target_centers = torch.stack(t_centers, dim=0)

        # --- one-to-one match & MSE ---
        matched_src, matched_tgt = self._match_centers_one_to_one(
            source_centers, target_centers
        )
        if matched_src is None:
            return None
        return F.mse_loss(matched_src, matched_tgt)

    @torch.no_grad()
    def _get_target_pseudo_labels(self, features: torch.Tensor) -> torch.Tensor:
        centers_data = self._target_class_centers_from_buffer()
        if centers_data is None:
            return features.new_full((features.shape[0],), -1, dtype=torch.long)

        centers, center_labels = centers_data
        if centers.shape[0] == 0:
            return features.new_full((features.shape[0],), -1, dtype=torch.long)

        norm_features = F.normalize(features, dim=1)
        norm_centers = F.normalize(centers, dim=1)
        dists = torch.cdist(norm_features, norm_centers, p=2)
        assign_idx = dists.argmin(dim=1)
        return center_labels[assign_idx]

    def _source_center_contrast_loss(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> Optional[torch.Tensor]:
        valid = (labels >= 0) & (labels < self.num_classes)
        if not valid.any():
            return None

        features = features[valid]
        labels = labels[valid]
        center_mask = self.source_class_counts > 0
        if center_mask.sum().item() < 2:
            return None

        centers = self.source_class_centers[center_mask]
        center_ids = torch.nonzero(center_mask, as_tuple=False).squeeze(1)
        label_map = {int(k.item()): i for i, k in enumerate(center_ids)}

        mapped_labels = torch.tensor(
            [label_map[int(k.item())] for k in labels],
            device=labels.device,
            dtype=torch.long,
        )
        return self._distance_ratio_loss(features, centers, mapped_labels)

    def _target_cluster_contrast_loss(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> Optional[torch.Tensor]:
        valid = (labels >= 0) & (labels < self.num_classes)
        if not valid.any():
            return None

        feats = features[valid]
        feat_labels = labels[valid]
        target_centers = self._target_class_centers_from_buffer()
        if target_centers is None:
            return None

        centers, center_labels = target_centers
        if centers.shape[0] < 2:
            return None

        center_map = {int(k.item()): i for i, k in enumerate(center_labels)}
        keep_mask = torch.tensor(
            [int(k.item()) in center_map for k in feat_labels],
            device=feat_labels.device,
            dtype=torch.bool,
        )
        if not keep_mask.any():
            return None

        feats = feats[keep_mask]
        feat_labels = feat_labels[keep_mask]
        mapped_labels = torch.tensor(
            [center_map[int(k.item())] for k in feat_labels],
            device=feat_labels.device,
            dtype=torch.long,
        )
        return self._distance_ratio_loss(feats, centers, mapped_labels)

    def _distance_ratio_loss(
        self, features: torch.Tensor, centers: torch.Tensor, labels: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if centers.shape[0] < 2:
            return None

        features = F.normalize(features, dim=1)
        centers = F.normalize(centers, dim=1)
        diffs = features[:, None, :] - centers[None, :, :]
        dists = (diffs * diffs).sum(dim=2)
        pos = dists.gather(1, labels.view(-1, 1)).squeeze(1)
        neg_sum = dists.sum(dim=1) - pos
        loss = pos / (neg_sum + self.contrastive_eps)
        return loss.mean()

    def _target_class_centers_from_buffer(
        self,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        centers = []
        center_labels = []
        for cls_id in range(self.num_classes):
            count = int(self.target_buffer_counts[cls_id].item())
            if count == 0:
                continue
            cls_feats = self.target_feature_buffer[cls_id, :count]
            centers.append(cls_feats.mean(dim=0))
            center_labels.append(cls_id)

        if not centers:
            return None
        return torch.stack(centers, dim=0), torch.tensor(
            center_labels, device=self.target_feature_buffer.device, dtype=torch.long
        )

    def _match_centers_one_to_one(
        self, source_centers: torch.Tensor, target_centers: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if source_centers.shape[0] == 0 or target_centers.shape[0] == 0:
            return None, None

        dists = torch.cdist(source_centers, target_centers, p=2)
        if not torch.isfinite(dists).all():
            finite = dists[torch.isfinite(dists)]
            penalty = (
                finite.max().item() * 10.0 + 1.0 if finite.numel() > 0 else 1e6
            )
            dists = torch.where(
                torch.isfinite(dists), dists, torch.full_like(dists, penalty)
            )

        # Global optimal one-to-one assignment on pairwise center distance.
        row_ind, col_ind = linear_sum_assignment(dists.detach().cpu().numpy())
        if len(row_ind) == 0:
            return None, None
        row_ind_t = torch.as_tensor(row_ind, device=source_centers.device, dtype=torch.long)
        col_ind_t = torch.as_tensor(col_ind, device=target_centers.device, dtype=torch.long)
        return source_centers[row_ind_t], target_centers[col_ind_t]

    def _dbscan(
        self, features: torch.Tensor, eps: float, min_samples: int
    ) -> Tuple[torch.Tensor, int]:
        num_points = features.shape[0]
        if num_points == 0:
            return features.new_full((0,), -1, dtype=torch.long), 0

        dists = torch.cdist(features, features, p=2)
        labels = features.new_full((num_points,), -1, dtype=torch.long)
        visited = torch.zeros(num_points, device=features.device, dtype=torch.bool)
        cluster_id = 0

        for i in range(num_points):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = torch.nonzero(dists[i] <= eps, as_tuple=False).squeeze(1)
            if neighbors.numel() < min_samples:
                labels[i] = -1
                continue

            labels[i] = cluster_id
            seeds = neighbors.tolist()
            idx = 0
            while idx < len(seeds):
                point = seeds[idx]
                if not visited[point]:
                    visited[point] = True
                    point_neighbors = torch.nonzero(
                        dists[point] <= eps, as_tuple=False
                    ).squeeze(1)
                    if point_neighbors.numel() >= min_samples:
                        for n in point_neighbors.tolist():
                            if n not in seeds:
                                seeds.append(n)
                if labels[point] == -1:
                    labels[point] = cluster_id
                idx += 1

            cluster_id += 1

        return labels, cluster_id

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], branch: str = ""
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        storage = get_event_storage()
        storage.put_scalar(
            "roi_head/num_target_fg_samples_" + branch, np.mean(num_fg_samples)
        )
        storage.put_scalar(
            "roi_head/num_target_bg_samples_" + branch, np.mean(num_bg_samples)
        )

        return proposals_with_gt
