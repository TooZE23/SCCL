# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from detectron2.config import CfgNode

from detectron2.solver.lr_scheduler import WarmupCosineLR, WarmupMultiStepLR
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.env import TORCH_VERSION as _TORCH_VERSION
from .lr_scheduler import WarmupTwoStageMultiStepLR

from typing import List, Tuple


def _build_custom_optimizer(
    cfg: CfgNode,
    params: List[torch.nn.Parameter],
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    betas: Tuple[float, float],
) -> torch.optim.Optimizer:
    optimizer_name = str(optimizer_name).upper()
    if optimizer_name == "ADAM":
        adam_args = {
            "params": params,
            "lr": lr,
            "betas": tuple(betas),
            "weight_decay": weight_decay,
        }
        if _TORCH_VERSION >= (1, 12):
            adam_args["foreach"] = True
        optimizer = torch.optim.Adam(**adam_args)
    elif optimizer_name == "SGD":
        sgd_args = {
            "params": params,
            "lr": lr,
            "momentum": cfg.SOLVER.MOMENTUM,
            "nesterov": cfg.SOLVER.NESTEROV,
            "weight_decay": weight_decay,
        }
        if _TORCH_VERSION >= (1, 12):
            sgd_args["foreach"] = True
        optimizer = torch.optim.SGD(**sgd_args)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_name}")

    return maybe_add_gradient_clipping(cfg, optimizer)


def build_optimizer(
    cfg: CfgNode, model: torch.nn.Module
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer]:
    """
    Build three optimizers:
    - optimizer_main: parameters except projection heads / shared decoder
    - optimizer_proj: projection heads only
    - optimizer_decoder: shared decoder only
    """
    proj_target_modules = tuple(cfg.SOLVER.PROJ_TARGET_MODULES)
    decoder_target_modules = tuple(cfg.SOLVER.DECODER_TARGET_MODULES)

    proj_params = []
    decoder_params = []
    main_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        in_proj = any(module_name in name for module_name in proj_target_modules)
        in_decoder = any(module_name in name for module_name in decoder_target_modules)

        if in_proj and in_decoder:
            raise ValueError(
                f"Parameter '{name}' matches both projection and decoder module patterns."
            )

        if in_proj:
            proj_params.append(param)
        elif in_decoder:
            decoder_params.append(param)
        else:
            main_params.append(param)

    if len(proj_params) == 0:
        raise ValueError(
            "No projection-head parameters found. Check SOLVER.PROJ_TARGET_MODULES: "
            f"{proj_target_modules}"
        )
    if len(decoder_params) == 0:
        raise ValueError(
            "No decoder parameters found. Check SOLVER.DECODER_TARGET_MODULES: "
            f"{decoder_target_modules}"
        )

    # Main optimizer (kept as SGD by default)
    sgd_args_main = {
        "params": main_params,
        "lr": cfg.SOLVER.BASE_LR,
        "momentum": cfg.SOLVER.MOMENTUM,
        "nesterov": cfg.SOLVER.NESTEROV,
        "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
    }
    if _TORCH_VERSION >= (1, 12):
        sgd_args_main["foreach"] = True
    optimizer_main = maybe_add_gradient_clipping(cfg, torch.optim.SGD(**sgd_args_main))

    optimizer_proj = _build_custom_optimizer(
        cfg=cfg,
        params=proj_params,
        optimizer_name=cfg.SOLVER.PROJ_OPTIMIZER,
        lr=cfg.SOLVER.PROJ_LR,
        weight_decay=cfg.SOLVER.PROJ_WEIGHT_DECAY,
        betas=tuple(cfg.SOLVER.PROJ_BETAS),
    )

    optimizer_decoder = _build_custom_optimizer(
        cfg=cfg,
        params=decoder_params,
        optimizer_name=cfg.SOLVER.DECODER_OPTIMIZER,
        lr=cfg.SOLVER.DECODER_LR,
        weight_decay=cfg.SOLVER.DECODER_WEIGHT_DECAY,
        betas=tuple(cfg.SOLVER.DECODER_BETAS),
    )

    return optimizer_main, optimizer_proj, optimizer_decoder

def build_lr_scheduler(
    cfg: CfgNode, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
    name = cfg.SOLVER.LR_SCHEDULER_NAME
    if name == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif name == "WarmupCosineLR":
        return WarmupCosineLR(
            optimizer,
            cfg.SOLVER.MAX_ITER,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif name == "WarmupTwoStageMultiStepLR":
        return WarmupTwoStageMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            factor_list=cfg.SOLVER.FACTOR_LIST, 
            gamma=cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))
