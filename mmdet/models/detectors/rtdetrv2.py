# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn
from mmdet.registry import MODELS
from ..layers import RTDETRHybridEncoder, RTDETRTransformerDecoderV2
from .rtdetr import RTDETR


@MODELS.register_module()
class RTDETRV2(RTDETR):
    r"""Implementation of `RT-DETRv2: Improved Baseline with Bag-of-Freebies
    for Real-Time Detection Transformer <https://arxiv.org/pdf/2407.17140>`_

    Code is modified from the `official github repo
    <https://github.com/lyuwenyu/RT-DETR>`_.
    """
    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.encoder = RTDETRHybridEncoder(**self.encoder)
        self.decoder = RTDETRTransformerDecoderV2(**self.decoder)
        self.embed_dims = self.decoder.embed_dims
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
