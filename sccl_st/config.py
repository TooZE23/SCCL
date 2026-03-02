# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_sccl_config(cfg):
    """
    Add config for semisupnet.
    """
    _C = cfg
    _C.TEST.VAL_LOSS = True

    _C.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0
    _C.MODEL.RPN.LOSS = "CrossEntropy"
    _C.MODEL.ROI_HEADS.LOSS = "CrossEntropy"
    _C.MODEL.BACKBONE.VGG16_IMAGENET_PRETRAINED = True
    _C.MODEL.BACKBONE.VGG16_IMAGENET_WEIGHTS = ""

    _C.SOLVER.IMG_PER_BATCH_LABEL = 1
    _C.SOLVER.IMG_PER_BATCH_UNLABEL = 1
    _C.SOLVER.FACTOR_LIST = (1,)
    _C.SOLVER.PROJ_OPTIMIZER = "ADAM"
    _C.SOLVER.PROJ_LR = 1e-3
    _C.SOLVER.PROJ_WEIGHT_DECAY = _C.SOLVER.WEIGHT_DECAY
    _C.SOLVER.PROJ_BETAS = (0.9, 0.999)
    _C.SOLVER.PROJ_TARGET_MODULES = ("projection_head1", "projection_head2")
    _C.SOLVER.DECODER_OPTIMIZER = "ADAM"
    _C.SOLVER.DECODER_LR = 1e-3
    _C.SOLVER.DECODER_WEIGHT_DECAY = _C.SOLVER.WEIGHT_DECAY
    _C.SOLVER.DECODER_BETAS = (0.9, 0.999)
    _C.SOLVER.DECODER_TARGET_MODULES = ("shared_decoder",)

    _C.DATASETS.TRAIN_LABEL = ("coco_2017_train",)
    _C.DATASETS.TRAIN_UNLABEL = ("coco_2017_train",)
    _C.DATASETS.CROSS_DATASET = True
    _C.TEST.EVALUATOR = "COCOeval"

    _C.SEMISUPNET = CN()

    # Output dimension of the MLP projector after `res5` block
    _C.SEMISUPNET.MLP_DIM = 128

    # Semi-supervised training
    _C.SEMISUPNET.Trainer = "scclteacher"
    _C.SEMISUPNET.BBOX_THRESHOLD = 0.7
    _C.SEMISUPNET.PSEUDO_BBOX_SAMPLE = "thresholding"
    _C.SEMISUPNET.TEACHER_UPDATE_ITER = 1
    _C.SEMISUPNET.BURN_UP_STEP = 12000
    _C.SEMISUPNET.EMA_KEEP_RATE = 0.0
    _C.SEMISUPNET.UNSUP_LOSS_WEIGHT = 4.0
    _C.SEMISUPNET.SUP_LOSS_WEIGHT = 0.5
    _C.SEMISUPNET.LOSS_WEIGHT_TYPE = "standard"
    _C.SEMISUPNET.DIS_TYPE = "res4"
    _C.SEMISUPNET.DIS_LOSS_WEIGHT = 0.1
    _C.SEMISUPNET.ITERS_PER_EPOCH = 743
    _C.SEMISUPNET.GRAM_HOOK = ["res3", "res4"]  # backbone feature levels for Gram-style contrast
    _C.SEMISUPNET.SHUFFLE_CKPT = ""  # path to pretrained shuffle classifier checkpoint
    _C.SEMISUPNET.DS_CHANNEL = 128   # domain-specific encoder channel dimension
    _C.TARGET_CONTRAST_START_EPOCH = 8
    # dataloader
    # supervision level
    _C.DATALOADER.SUP_PERCENT = 100.0  # 5 = 5% dataset as labeled set
    _C.DATALOADER.RANDOM_DATA_SEED = 0  # random seed to read data
    _C.DATALOADER.RANDOM_DATA_SEED_PATH = "dataseed/COCO_supervision.txt"

    _C.EMAMODEL = CN()
    _C.EMAMODEL.SUP_CONSIST = True
