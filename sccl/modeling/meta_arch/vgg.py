# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torch.nn as nn
import torch
from typing import Union, List, Dict, cast
import detectron2.utils.comm as comm
from detectron2.modeling.backbone import (
    Backbone,
    BACKBONE_REGISTRY
)
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool


logger = logging.getLogger(__name__)


def _emit_vgg_init_log(message: str, level: int = logging.INFO) -> None:
    tagged = f"[VGGInit] {message}"
    logger.log(level, tagged)
    # Make initialization status always visible in training stdout.
    if comm.is_main_process():
        print(tagged, flush=True)

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs: Dict[str, List[Union[str, int]]] = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class vgg_backbone(Backbone):
    """
    Backbone (bottom-up) for FBNet.

    Hierarchy:
        trunk0:
            xif0_0
            xif0_1
            ...
        trunk1:
            xif1_0
            xif1_1
            ...
        ...

    Output features:
        The outputs from each "stage", i.e. trunkX.
    """

    def __init__(self, cfg):
        super().__init__()

        self.vgg = make_layers(cfgs['vgg16'],batch_norm=True)

        self._initialize_weights()
        self._load_imagenet_pretrained(cfg)
        # self.stage_names_index = {'vgg1':3, 'vgg2':8 , 'vgg3':15, 'vgg4':22, 'vgg5':29}
        _out_feature_channels = [64, 128, 256, 512, 512]
        _out_feature_strides = [2, 4, 8, 16, 32]
        # stages, shape_specs = build_fbnet(
        #     cfg,
        #     name="trunk",
        #     in_channels=cfg.MODEL.FBNET_V2.STEM_IN_CHANNELS
        # )

        # nn.Sequential(*list(self.vgg.features._modules.values())[:14])

        self.stages = [nn.Sequential(*list(self.vgg._modules.values())[0:7]),\
                    nn.Sequential(*list(self.vgg._modules.values())[7:14]),\
                    nn.Sequential(*list(self.vgg._modules.values())[14:24]),\
                    nn.Sequential(*list(self.vgg._modules.values())[24:34]),\
                    nn.Sequential(*list(self.vgg._modules.values())[34:]),]
        self._out_feature_channels = {}
        self._out_feature_strides = {}
        self._stage_names = []

        for i, stage in enumerate(self.stages):
            name = "vgg{}".format(i)
            self.add_module(name, stage)
            self._stage_names.append(name)
            self._out_feature_channels[name] = _out_feature_channels[i]
            self._out_feature_strides[name] = _out_feature_strides[i]

        self._out_features = self._stage_names
        self._size_divisibility = 32

        del self.vgg

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        features = {}
        for name, stage in zip(self._stage_names, self.stages):
            x = stage(x)
            # if name in self._out_features:
            #     outputs[name] = x
            features[name] = x
        # import pdb
        # pdb.set_trace()

        return features

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _extract_vgg_state_dict(self, state_dict):
        model_keys = set(self.vgg.state_dict().keys())
        extracted = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[len("module."):]
            if new_key.startswith("backbone."):
                new_key = new_key[len("backbone."):]
            if new_key.startswith("vgg."):
                new_key = new_key[len("vgg."):]
            if new_key.startswith("features."):
                new_key = new_key[len("features."):]
            if new_key in model_keys:
                extracted[new_key] = value
        return extracted

    def _load_imagenet_pretrained(self, cfg) -> None:
        if not cfg.MODEL.BACKBONE.VGG16_IMAGENET_PRETRAINED:
            _emit_vgg_init_log(
                "VGG16_IMAGENET_PRETRAINED=False, using random initialization.",
                level=logging.WARNING,
            )
            return

        weights_path = cfg.MODEL.BACKBONE.VGG16_IMAGENET_WEIGHTS
        try:
            if weights_path:
                checkpoint = torch.load(weights_path, map_location="cpu")
                state_dict = checkpoint
                if isinstance(checkpoint, dict):
                    if "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    elif "model" in checkpoint:
                        state_dict = checkpoint["model"]

                state_dict = self._extract_vgg_state_dict(state_dict)
                if len(state_dict) == 0:
                    raise ValueError(f"No VGG backbone weights matched in: {weights_path}")

                missing_keys, unexpected_keys = self.vgg.load_state_dict(state_dict, strict=False)
                _emit_vgg_init_log(
                    "Loaded VGG16 ImageNet weights from "
                    f"{weights_path} (matched={len(state_dict)}, "
                    f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)})."
                )
                return

            from torchvision.models import vgg16_bn
            try:
                from torchvision.models import VGG16_BN_Weights
                tv_model = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
            except (ImportError, AttributeError, TypeError):
                tv_model = vgg16_bn(pretrained=True)

            missing_keys, unexpected_keys = self.vgg.load_state_dict(
                tv_model.features.state_dict(), strict=False
            )
            _emit_vgg_init_log(
                "Loaded torchvision VGG16_BN ImageNet weights "
                f"(missing={len(missing_keys)}, unexpected={len(unexpected_keys)})."
            )
        except Exception as exc:
            _emit_vgg_init_log(
                "Failed to load ImageNet VGG16_BN pretrained weights, "
                f"fallback to random initialization: {exc}",
                level=logging.WARNING,
            )


@BACKBONE_REGISTRY.register() #already register in baseline model
def build_vgg_backbone(cfg, _):
    return vgg_backbone(cfg)


@BACKBONE_REGISTRY.register() #already register in baseline model
def build_vgg_fpn_backbone(cfg, _):
    bottom_up = vgg_backbone(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        # fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    # return backbone

    return backbone
