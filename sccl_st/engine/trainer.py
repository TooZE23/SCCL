# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import copy
import time
import logging
import torch
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import (
    verify_results,
    DatasetEvaluators,
    CityscapesInstanceEvaluator,
)

from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.data import MetadataCatalog

from sccl_st.data.build import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_no_crop,
    build_detection_semisup_train_loader_two_crops,
)
from sccl_st.data.dataset_mapper import DatasetMapperTwoCropSeparate
from sccl_st.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from sccl_st.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from sccl_st.engine.hooks import LossEvalHook
from sccl_st.solver.build import build_lr_scheduler, build_optimizer
from sccl_st.evaluation import PascalVOCDetectionEvaluator, COCOEvaluator

from .probe import OpenMatchTrainerProbe


# Supervised-only Trainer
class BaselineTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        # Expose model and optimizer on the trainer instance
        self.model = model
        self.optimizer = optimizer

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def run_step(self):
        self._trainer.iter = self.iter

        assert self.model.training, "[BaselineTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        record_dict, _, _, _ = self.model(data, branch="supervised")

        num_gt_bbox = 0.0
        for element in data:
            num_gt_bbox += len(element["instances"])
        num_gt_bbox = num_gt_bbox / len(data)
        record_dict["bbox_num/gt_bboxes"] = num_gt_bbox

        loss_dict = {}
        for key in record_dict.keys():
            if key[:4] == "loss" and key[-3:] != "val":
                loss_dict[key] = record_dict[key]

        losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "cityscapes_instance":
            return CityscapesInstanceEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(dataset_name, target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_semisup_train_loader(cfg, mapper=None)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        """
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 0:
                self.storage.put_scalars(**metrics_dict)


# =====================================================
# SCCL + Teacher-Student Fused Trainer
# =====================================================
class SCCLTeacherTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Fused trainer combining:
        - SCCL domain adaptation (style contrast, domain-specific encoders, shared decoder)
        - Teacher-Student pseudo-label framework (EMA teacher, pseudo-label thresholding)
        
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        # Build student model (SCCL DAobjGeneralizedRCNN)
        model = self.build_model(cfg)
        # Build three optimizers: main / projection heads / shared decoder
        optimizer, optimizer_proj, optimizer_decoder = self.build_optimizer(cfg, model)

        # Build teacher model (same architecture, used for pseudo-label generation)
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        # Strip SCCL-specific modules from teacher to save GPU memory.
        # Teacher only needs backbone + RPN + ROI heads for pseudo-label generation.
        self._strip_teacher_sccl_modules(self.model_teacher)
        # Freeze teacher parameters (updated via EMA, not gradient)
        for param in self.model_teacher.parameters():
            param.requires_grad = False
        # Teacher is used only for pseudo-label inference.
        self.model_teacher.eval()

        # Wrap student with DDP for multi-GPU
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.model = model
        self.optimizer = optimizer
    
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # Projection/decoder optimizers and schedulers
        self.optimizer_proj = optimizer_proj
        self.scheduler_proj = self.build_lr_scheduler(cfg, optimizer_proj)
        self.optimizer_decoder = optimizer_decoder
        self.scheduler_decoder = self.build_lr_scheduler(cfg, optimizer_decoder)

        # Ensemble model for unified checkpoint save/load
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
            optimizer_proj=optimizer_proj,
            scheduler_proj=self.scheduler_proj,
            optimizer_decoder=optimizer_decoder,
            scheduler_decoder=self.scheduler_decoder,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self._teacher_initialized = False

        self.probe = OpenMatchTrainerProbe(cfg)
        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # Full TS checkpoint includes teacher weights/state already.
            self._teacher_initialized = True
        else:
            self._teacher_initialized = False
        if isinstance(self.model, DistributedDataParallel):
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "cityscapes_instance":
            if cfg.MODEL.MASK_ON:
                return CityscapesInstanceEvaluator(dataset_name)
            return COCOEvaluator(dataset_name, output_dir=output_folder)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(
                dataset_name,
                target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"],
            )
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        return build_optimizer(cfg, model)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
        if comm.is_main_process() and hasattr(self, "_last_eval_results"):
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    # =====================================================
    # ================== Pseudo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        return new_proposal_inst

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unknown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data

    def get_label(self, label_data):
        label_list = []
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                label_list.append(copy.deepcopy(label_datum["instances"]))
        return label_list

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[SCCLTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # DataLoader yields 4 groups: label_q(strong), label_k(weak), unlabel_q(strong), unlabel_k(weak)
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
        data_time = time.perf_counter() - start

        # Set current epoch for ROI head target contrastive logic
        current_epoch = self.iter // self.cfg.SEMISUPNET.ITERS_PER_EPOCH
        model_ref = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        model_ref.roi_heads.set_current_epoch(current_epoch)
        if current_epoch == self.cfg.TARGET_CONTRAST_START_EPOCH:
            target_flag = "initial"
        elif current_epoch > self.cfg.TARGET_CONTRAST_START_EPOCH:
            target_flag = "final"
        else:
            target_flag = None

        # =====================================================
        # BURN-IN: supervised training only
        # =====================================================
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:

            # Merge strong + weak augmented labeled data
            all_label_data = label_data_q + label_data_k
            record_dict, _, _, _ = self.model(
                all_label_data, branch="supervised"
            )

            # All losses with weight 1
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())

        # =====================================================
        # POST BURN-IN: Teacher-Student + SCCL domain adaptation
        # =====================================================
        else:
            record_dict = {}

            # =============================================
            # Alternating optimization
            # =============================================
            if self.iter % 3 != 2:
                # --------------------------------------------------
                # Main network optimization (iter % 3 == 0 or 1)
                # --------------------------------------------------
                if not self._teacher_initialized:
                    self._update_teacher_model(keep_rate=0.00)
                    self._teacher_initialized = True
                elif (
                    self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
                ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                    self._update_teacher_model(
                        keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE
                    )

                # Remove labels from unlabeled data before pseudo-labeling.
                unlabel_data_q = self.remove_label(unlabel_data_q)
                unlabel_data_k = self.remove_label(unlabel_data_k)

                # ---- Step 1: Teacher generates pseudo-labels ----
                with torch.no_grad():
                    self.model_teacher.eval()
                    (
                        _,
                        proposals_rpn_unsup_k,
                        proposals_roih_unsup_k,
                        _,
                    ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

                # ---- Step 2: Threshold pseudo-labels ----
                cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

                joint_proposal_dict = {}
                joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k

                pesudo_proposals_rpn_unsup_k, nun_pseudo_bbox_rpn = self.process_pseudo_label(
                    proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
                )
                joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k

                pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                    proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
                )
                joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k

                # ---- Step 3: Add pseudo-labels to unlabeled data ----
                unlabel_data_q = self.add_label(
                    unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
                )
                unlabel_data_k = self.add_label(
                    unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
                )

                # (A) Supervised detection on ALL labeled data (strong + weak)
                all_label_data = label_data_q + label_data_k
                record_all_label_data, _, _, _ = self.model(
                    all_label_data, branch="supervised"
                )
                record_dict.update(record_all_label_data)

                # (B) Pseudo-label detection on strongly augmented unlabeled data
                all_unlabel_data = unlabel_data_q
                record_all_unlabel_data, _, _, _ = self.model(
                    all_unlabel_data, branch="supervised_target"
                )
                # Rename keys with _pseudo suffix
                new_record_all_unlabel_data = {}
                for key in record_all_unlabel_data.keys():
                    new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[key]
                record_dict.update(new_record_all_unlabel_data)

                # (C) SCCL domain adaptation: pack source(weak) + target(weak)
                for i_index in range(len(unlabel_data_k)):
                    for k, v in unlabel_data_k[i_index].items():
                        label_data_k[i_index][k + "_unlabeled"] = v

                # main branch now returns only SCCL-specific losses
                # (detection losses skipped via source_contrast_only branch)
                record_sccl = self.model(
                    label_data_k, branch="main", target_flag=target_flag
                )
                
                for key, val in record_sccl.items():
                    if key.startswith("loss"):
                        record_dict[key] = val

                # ---- Weight losses ----
                loss_dict = {}
                for key in record_dict.keys():
                    if key.startswith("loss"):
                        if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                            # Pseudo bbox regression loss → 0
                            loss_dict[key] = record_dict[key] * 0
                        elif key.endswith("_pseudo"):
                            # Unsupervised pseudo-label loss
                            loss_dict[key] = (
                                record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                            )
                        else:
                            # Supervised + SCCL domain adaptation losses
                            loss_dict[key] = record_dict[key] * 1

                losses = sum(loss_dict.values())

            else:
                # --------------------------------------------------
                # Projection head optimization (iter % 3 == 2)
                # --------------------------------------------------
                for i_index in range(len(unlabel_data_k)):
                    for k, v in unlabel_data_k[i_index].items():
                        label_data_k[i_index][k + "_unlabeled"] = v

                record_dict = self.model(
                    label_data_k, branch="projection", target_flag=target_flag
                )

                loss_dict = {}
                for key in record_dict.keys():
                    if key.startswith("loss"):
                        loss_dict[key] = record_dict[key] * 1

                losses = sum(loss_dict.values())

        # ---- Metrics & backward ----
        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        is_post_burn = self.iter >= self.cfg.SEMISUPNET.BURN_UP_STEP
        is_projection_step = is_post_burn and self.iter % 3 == 2

        if is_projection_step:
            self.optimizer_proj.zero_grad()
            losses.backward()
            self.optimizer_proj.step()
            self.scheduler_proj.step()
        else:
            self.optimizer.zero_grad()
            if is_post_burn:
                self.optimizer_decoder.zero_grad()
            losses.backward()
            self.optimizer.step()
            if is_post_burn:
                self.optimizer_decoder.step()
                self.scheduler_decoder.step()

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 0:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.9996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict, strict=False)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict(), strict=False)

    @staticmethod
    def _strip_teacher_sccl_modules(model_teacher):
        """Remove SCCL-specific modules from teacher to save GPU memory.

        Teacher only uses backbone + RPN + ROI heads for pseudo-label
        generation (unsup_data_weak branch). The following modules are
        never accessed by teacher and can be safely deleted:
          - projection_head1 / projection_head2  (style contrast)
          - domain_specific_encoder_s / _t        (domain encoding)
          - shared_decoder                        (reconstruction)
          - ds_projector                          (domain-specific contrastive)
          - shuffle_classifier                    (frozen classifier)
        """
        sccl_module_names = [
            "projection_head1",
            "projection_head2",
            "domain_specific_encoder_s",
            "domain_specific_encoder_t",
            "shared_decoder",
            "ds_projector",
            "shuffle_classifier",
        ]
        for name in sccl_module_names:
            if hasattr(model_teacher, name):
                delattr(model_teacher, name)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model_teacher
            )
            # Keep detectron2-compatible key for train()-time verify/return.
            self._last_eval_results = self._last_eval_results_teacher
            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results_teacher))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret
