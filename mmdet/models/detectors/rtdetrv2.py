# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import BaseModule, PretrainedInit
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

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        # for tuning
        if self.init_cfg is not None:
            init_cfgs = self.init_cfg
            if isinstance(init_cfgs, dict):
                init_cfgs = [self.init_cfg]
            for init_cfg in init_cfgs:
                assert isinstance(init_cfg, dict)
                if (init_cfg['type'] == 'Pretrained'
                        or init_cfg['type'] is PretrainedInit):
                    BaseModule.init_weights(self)
                    return

        super().init_weights()
