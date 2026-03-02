# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import numpy as np
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import build_backbone, Backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

from .blocks import shuffle_classifier, DomainSpecificEncoder, SharedDecoder, ProjectionHead1, ProjectionHead2, ContrastiveProjector

def compute_gram_matrix(features):
    """
    计算Gram matrix
    Args:
        features: shape [B, C, H, W]
    Returns:
        gram: shape [B, C, C]
    """
    B, C, H, W = features.shape
    features = features.view(B, C, H * W)  # [B, C, H*W]
    gram = torch.bmm(features, features.transpose(1, 2))  # [B, C, C]
    # 归一化
    gram = gram / (C * H * W) # N^l*M^l
    return gram



@META_ARCH_REGISTRY.register()
class DAobjGeneralizedRCNN(GeneralizedRCNN):

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        dis_type: str,
        gram_hook: list,
        shuffle_ckpt: str,
        ds_channel: int = 128,
        temperature: float = 0.15,
        # dis_loss_weight: float = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__(
            backbone=backbone,
            proposal_generator=proposal_generator,
            roi_heads=roi_heads,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            input_format=input_format,
            vis_period=vis_period,
        )
        
        self.dis_type = dis_type
        self.gram_hook = gram_hook
        self.ds_channel = ds_channel
        self.temperature = temperature
        self.domain_specific_encoder_s = DomainSpecificEncoder(self.backbone._out_feature_channels[self.gram_hook[1]], self.ds_channel)
        self.domain_specific_encoder_t = DomainSpecificEncoder(self.backbone._out_feature_channels[self.gram_hook[1]],self.ds_channel)
        self.shuffle_classifier = self.build_shuffle_classifier(shuffle_ckpt)
    
        # 初始化两个可训练的projection heads
        # 假设特征通道数为C,Gram matrix展平后为C*C维
        feature_channels1 = self.backbone._out_feature_channels[self.gram_hook[0]]
        feature_channels2 = self.backbone._out_feature_channels[self.gram_hook[1]]
        gram_dim1 = feature_channels1 * feature_channels1
        gram_dim2 = feature_channels2 * feature_channels2

        self.projection_head1 = ProjectionHead1(
            input_dim=gram_dim1,
            hidden_dim=256,
            output_dim=64
        )
        
        self.projection_head2 = ProjectionHead2(
            input_dim=gram_dim2,
            hidden_dim1=512,
            hidden_dim2=256,
            output_dim=64
        )

        decoder_in_channels = self.backbone._out_feature_channels[self.dis_type] + self.ds_channel
        decoder_out_channels = self.backbone._out_feature_channels[self.gram_hook[1]]
        
        # Compute upsample factor dynamically: dis_type stride / gram_hook[1] stride
        # e.g. ResNet: res4(16) / res1(4) = 4; VGG: vgg4(32) / vgg1(4) = 8
        dis_stride = self.backbone._out_feature_strides[self.dis_type]
        hook_stride = self.backbone._out_feature_strides[self.gram_hook[1]]
        upsample_factor = dis_stride // hook_stride
        
        self.shared_decoder = SharedDecoder(
            in_channels=decoder_in_channels,
            out_channels=decoder_out_channels,
            upsample_factor=upsample_factor,
        )

        self.ds_projector = ContrastiveProjector(self.ds_channel,2048,128)
        self.hinge_embedding_loss = nn.HingeEmbeddingLoss(margin=0.5)

    def build_shuffle_classifier(self, shuffle_ckpt):
        classifier = shuffle_classifier(
            self.backbone._out_feature_channels[self.dis_type] + 
            self.ds_channel
        )
        
        if shuffle_ckpt and os.path.exists(shuffle_ckpt):
            checkpoint = torch.load(shuffle_ckpt, map_location='cuda')
            # 根据checkpoint的格式选择合适的加载方式
            if 'model' in checkpoint:
                classifier.load_state_dict(checkpoint['model'])
            elif 'state_dict' in checkpoint:
                classifier.load_state_dict(checkpoint['state_dict'])
            else:
                classifier.load_state_dict(checkpoint)
            print(f"Loaded shuffle classifier weights from {shuffle_ckpt}")
        
        for param in classifier.parameters():
            param.requires_grad = False
        
        classifier.eval()
        
        return classifier

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "dis_type": cfg.SEMISUPNET.DIS_TYPE,
            "gram_hook": cfg.SEMISUPNET.GRAM_HOOK,
            "shuffle_ckpt": cfg.SEMISUPNET.SHUFFLE_CKPT,
            "ds_channel": cfg.SEMISUPNET.DS_CHANNEL,
            "temperature": 0.15,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    @contextmanager
    def _freeze_projection_heads(self):
        """Temporarily freeze projection heads while keeping input gradients."""
        modules = (self.projection_head1, self.projection_head2)
        requires_grad_cache = []
        for module in modules:
            module_cache = []
            for param in module.parameters():
                module_cache.append(param.requires_grad)
                param.requires_grad_(False)
            requires_grad_cache.append(module_cache)
        try:
            yield
        finally:
            for module, module_cache in zip(modules, requires_grad_cache):
                for param, require_grad in zip(module.parameters(), module_cache):
                    param.requires_grad_(require_grad)

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs, branch="main", target_flag=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        if branch == "sourceonly":
            # Baseline/source-only path: only labeled source images are expected.
            images_s = self.preprocess_image(batched_inputs)
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None

            losses = {}
            features_s = self.backbone(images_s.tensor)
            if self.proposal_generator is not None:
                proposals_s, proposal_losses_s = self.proposal_generator(images_s, features_s, gt_instances)
            else:
                assert "proposals" in batched_inputs[0]
                proposals_s = [x["proposals"].to(self.device) for x in batched_inputs]
                proposal_losses_s = {}
            
            _, detector_losses_s = self.roi_heads(images_s, features_s, proposals_s, gt_instances, branch="sourceonly")
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_s)
           

            losses.update(detector_losses_s)
            losses.update(proposal_losses_s)
            return losses

        source_label = 0
        target_label = 1

        images_s, images_t = self.preprocess_image_train(batched_inputs)
        assert len(images_s) == len(images_t), "Source and target batch size must be equal"
        N = len(images_s)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        losses = {}
        features_s = self.backbone(images_s.tensor)

        if branch == "projection":
            features_t = self.backbone(images_t.tensor)
            gram_s_1 = compute_gram_matrix(features_s[self.gram_hook[0]])
            gram_s_2 = compute_gram_matrix(features_s[self.gram_hook[1]])
            gram_t_1 = compute_gram_matrix(features_t[self.gram_hook[0]])
            gram_t_2 = compute_gram_matrix(features_t[self.gram_hook[1]])

            style_s_1 = self.projection_head1(gram_s_1.detach())
            style_t_1 = self.projection_head1(gram_t_1.detach())
            style_s_2 = self.projection_head2(gram_s_2.detach())
            style_t_2 = self.projection_head2(gram_t_2.detach())
            
            style_s_1 = F.normalize(style_s_1, dim=-1)
            style_t_1 = F.normalize(style_t_1, dim=-1)
            style_s_2 = F.normalize(style_s_2, dim=-1)
            style_t_2 = F.normalize(style_t_2, dim=-1)

            if N < 2:
                # Avoid empty positive pairs when per-GPU batch size is 1.
                losses["loss_hinge"] = (
                    style_s_1.sum() * 0.0
                    + style_t_1.sum() * 0.0
                    + style_s_2.sum() * 0.0
                    + style_t_2.sum() * 0.0
                )
                return losses
            
            sim_ss_1 = 1 - style_s_1 @ style_s_1.T
            mask_pos = ~torch.eye(sim_ss_1.size(0), dtype=torch.bool, device=style_s_1.device)
            # mask_neg = torch.triu(torch.ones(N, N, dtype=torch.bool, device=style_s_1.device))
            sim_tt_1 = 1 - style_t_1 @ style_t_1.T
            sim_tt_pos_1 = sim_tt_1[mask_pos] #1D向量，长度N*(N-1)
            sim_ss_pos_1 = sim_ss_1[mask_pos]
            sim_st_neg_1 = 1 - style_s_1 @ style_t_1.T

            sim_ss_2 = 1 - style_s_2 @ style_s_2.T
            sim_tt_2 = 1 - style_t_2 @ style_t_2.T
            sim_tt_pos_2 = sim_tt_2[mask_pos]
            sim_ss_pos_2 = sim_ss_2[mask_pos]
            sim_st_neg_2 = 1 - style_s_2 @ style_t_2.T  ##优化只留一个三角矩阵即可
            sim_st_neg_1 = sim_st_neg_1.reshape(-1)
            sim_st_neg_2 = sim_st_neg_2.reshape(-1)
            target_similar = torch.ones(sim_tt_pos_1.shape[0], device=style_s_2.device)
            target_dissimilar = -torch.ones(sim_st_neg_1.shape[0], device=style_s_2.device)
            loss_hinge = (
                self.hinge_embedding_loss(sim_tt_pos_1, target_similar)
                + self.hinge_embedding_loss(sim_ss_pos_1, target_similar)
                + self.hinge_embedding_loss(sim_tt_pos_2, target_similar)
                + self.hinge_embedding_loss(sim_ss_pos_2, target_similar)
                + self.hinge_embedding_loss(sim_st_neg_1, target_dissimilar)
                + self.hinge_embedding_loss(sim_st_neg_2, target_dissimilar)
            )
            losses["loss_hinge"] = loss_hinge
        
            return losses
        else:
            features_t = self.backbone(images_t.tensor)
            gram_s_1 = compute_gram_matrix(features_s[self.gram_hook[0]])
            gram_s_2 = compute_gram_matrix(features_s[self.gram_hook[1]])
            gram_t_1 = compute_gram_matrix(features_t[self.gram_hook[0]])
            gram_t_2 = compute_gram_matrix(features_t[self.gram_hook[1]])

            with self._freeze_projection_heads():
                style_s_1 = self.projection_head1(gram_s_1)
                style_t_1 = self.projection_head1(gram_t_1)
                style_s_2 = self.projection_head2(gram_s_2)
                style_t_2 = self.projection_head2(gram_t_2)
            
            loss_style = 1/4*F.mse_loss(style_s_1, style_t_1) + 1/4*F.mse_loss(style_s_2, style_t_2) 
            losses["loss_style"] = loss_style

            ds_features_s = self.domain_specific_encoder_s(features_s[self.gram_hook[1]], features_s[self.dis_type].shape[2:])
            ds_features_t = self.domain_specific_encoder_t(features_t[self.gram_hook[1]], features_t[self.dis_type].shape[2:])
            ds_s_proj = self.ds_projector(ds_features_s)
            ds_t_proj = self.ds_projector(ds_features_t)
            loss_ds_contrast, _, _ = self.contrastive_loss(ds_s_proj, ds_t_proj)
            losses["loss_ds_contrast"] = loss_ds_contrast

            # Build "ordered vs shuffled" pairs for shuffle-classification:
            # ds features keep source->target order, while shared features are
            # either kept in-order or randomly permuted.
            ds_concat = torch.cat([ds_features_s, ds_features_t], dim=0)
            feat_concat = torch.cat(
                [features_s[self.dis_type], features_t[self.dis_type]], dim=0
            )
            shuffle_perm = torch.randperm(feat_concat.shape[0], device=feat_concat.device)
            if feat_concat.shape[0] > 1 and torch.all(
                shuffle_perm == torch.arange(feat_concat.shape[0], device=feat_concat.device)
            ):
                shuffle_perm = torch.roll(shuffle_perm, shifts=1)
            feat_concat_shuffled = feat_concat[shuffle_perm]

            shuffle_input_ordered = torch.cat([ds_concat, feat_concat], dim=1)
            shuffle_input_shuffled = torch.cat([ds_concat, feat_concat_shuffled], dim=1)

            shuffle_cls_ordered = self.shuffle_classifier(shuffle_input_ordered)
            shuffle_cls_shuffled = self.shuffle_classifier(shuffle_input_shuffled)
            num_source = ds_features_s.shape[0]
            num_target = ds_features_t.shape[0]
            domain_labels = torch.cat(
                [
                    torch.full((num_source,), float(source_label), device=ds_concat.device),
                    torch.full((num_target,), float(target_label), device=ds_concat.device),
                ],
                dim=0,
            ).view(-1, 1, 1, 1)
            loss_shuffle_cls = (
                F.binary_cross_entropy_with_logits(
                    shuffle_cls_ordered,
                    domain_labels.expand_as(shuffle_cls_ordered),
                )
                + F.binary_cross_entropy_with_logits(
                    shuffle_cls_shuffled,
                    domain_labels.expand_as(shuffle_cls_shuffled),
                )
            )
            losses["loss_shuffle_cls"] = loss_shuffle_cls

            decoder_s = self.shared_decoder(torch.cat([ds_features_s, features_s[self.dis_type]], dim=1), target_size=features_s[self.gram_hook[1]].shape[2:])
            decoder_t = self.shared_decoder(torch.cat([ds_features_t, features_t[self.dis_type]], dim=1), target_size=features_t[self.gram_hook[1]].shape[2:])
            loss_decoder = F.l1_loss(decoder_s, features_s[self.gram_hook[1]]) + F.l1_loss(decoder_t, features_t[self.gram_hook[1]])
            losses["loss_decoder"] = loss_decoder

            ## source proposal and roi head loss
            if self.proposal_generator is not None:
                proposals_s, proposal_losses_s = self.proposal_generator(images_s, features_s, gt_instances)
            else:
                assert "proposals" in batched_inputs[0]
                proposals_s = [x["proposals"].to(self.device) for x in batched_inputs]
                proposal_losses_s = {}

            _, roi_losses_s = self.roi_heads(images_s, features_s, proposals_s, gt_instances, branch="source")
            losses.update(roi_losses_s)
            
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_s)
            
            ## target proposal and roi head loss (不计算目标域的检测损失，只计算对比损失等)
            if target_flag == "initial":
                if self.proposal_generator is not None:
                    proposals_t, _ = self.proposal_generator(images_t, features_t, branch="target")
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals_t = [x["proposals"].to(self.device) for x in batched_inputs]
                _, _ = self.roi_heads(images_t, features_t, proposals_t, None, branch="target")

            elif target_flag == "final":
                proposals_t, _ = self.proposal_generator(images_t, features_t, None, branch="target")
                _, roi_losses_t = self.roi_heads(images_t, features_t, proposals_t, None, branch="target")
                losses.update(roi_losses_t)

            losses.update(proposal_losses_s)
            return losses

    def contrastive_loss(self, u_s, u_t):
        """
        Args:
            u_s: source domain feature [B, D]
            u_t: target domain feature [B, D]
        Returns:
            loss: contrastive loss
        """
        N = u_s.shape[0]
        device = u_s.device
        if N < 2:
            zero = u_s.sum() * 0.0 + u_t.sum() * 0.0
            return zero, zero, zero
        
        # ========== 源域对比损失 ==========
        # 相似度矩阵
        sim_ss = torch.matmul(u_s, u_s.T) / self.temperature  # [N, N]
        sim_st = torch.matmul(u_s, u_t.T) / self.temperature  # [N, N]
        
        # 创建mask,排除对角线(自己和自己)
        mask_self = torch.eye(N, dtype=torch.bool, device=device)
        
        # 对于源域: 正样本是域内其他样本
        # 使用log-sum-exp技巧避免数值不稳定
        
        # 计算每个样本的损失
        loss_source_list = []
        for i in range(N):
            # 正样本logits (域内除了自己)
            pos_logits = sim_ss[i][~mask_self[i]]  # [N-1]
            
            # 负样本logits (所有目标域样本)
            neg_logits = sim_st[i]  # [N]
            
            # 对每个正样本计算损失
            for pos_logit in pos_logits:
                # log(exp(pos) / (exp(pos) + sum(exp(neg))))
                # = pos - log(exp(pos) + sum(exp(neg)))
                all_logits = torch.cat([pos_logit.unsqueeze(0), neg_logits])
                log_denominator = torch.logsumexp(all_logits, dim=0)
                loss = -pos_logit + log_denominator
                loss_source_list.append(loss)
        
        loss_source = torch.stack(loss_source_list).mean()
        
        # ========== 目标域对比损失 ==========
        sim_tt = torch.matmul(u_t, u_t.T) / self.temperature  # [N, N]
        sim_ts = torch.matmul(u_t, u_s.T) / self.temperature  # [N, N]
        
        loss_target_list = []
        for i in range(N):
            # 正样本logits (域内除了自己)
            pos_logits = sim_tt[i][~mask_self[i]]  # [N-1]
            
            # 负样本logits (所有源域样本)
            neg_logits = sim_ts[i]  # [N]
            
            for pos_logit in pos_logits:
                all_logits = torch.cat([pos_logit.unsqueeze(0), neg_logits])
                log_denominator = torch.logsumexp(all_logits, dim=0)
                loss = -pos_logit + log_denominator
                loss_target_list.append(loss)
        
        loss_target = torch.stack(loss_target_list).mean()
        
        # 总损失
        total_loss = loss_source + loss_target
        
        return total_loss, loss_source, loss_target


    def preprocess_image_train(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        images_t = [x["image_unlabeled"].to(self.device) for x in batched_inputs]
        images_t = [(x - self.pixel_mean) / self.pixel_std for x in images_t]
        images_t = ImageList.from_tensors(images_t, self.backbone.size_divisibility)

        return images, images_t

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

