# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from .vision_transformer import DinoVisionTransformer

__all__ = ['DinoVisionTransformer']

MODELS.register_module(module=DinoVisionTransformer)
