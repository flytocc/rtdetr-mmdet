# Copyright (c) OpenMMLab. All rights reserved.
import math
from copy import deepcopy
from torch import nn

from mmdet.registry import MODELS
from .deformable_detr import DeformableDETR, MultiScaleDeformableAttention
from .deim import DEIMDFINE
from ..layers import DEIMV2TransformerDecoder, MLP
from ..layers.transformer.dfine_layers import (
    LQE, Gate, MultiNumPointsMultiScaleDeformableAttention)


@MODELS.register_module()
class DEIMV2(DEIMDFINE):
    """Implementation of `Real-Time Object Detection Meets DINOv3
    <https://arxiv.org/abs/2509.20787>`_
    """

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        ref_act_cfg = self.decoder.pop('ref_act_cfg',
                                       dict(type='SiLU', inplace=True))
        ref_hidden_dim = self.decoder.pop('ref_hidden_dim', None)
        ref_num_layers = self.decoder.pop('ref_num_layers', 2)

        decoder_cfg = deepcopy(self.decoder)
        super()._init_layers()
        self.decoder = DEIMV2TransformerDecoder(**decoder_cfg)
        self.memory_trans_fc = nn.Identity()
        self.memory_trans_norm = nn.Identity()

        self.decoder.ref_point_head = MLP(
            4,
            ref_hidden_dim or self.decoder.embed_dims * 2,
            self.decoder.embed_dims,
            ref_num_layers,
            act_cfg=ref_act_cfg)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(DeformableDETR, self).init_weights()
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m,
                          (Gate, MultiNumPointsMultiScaleDeformableAttention,
                           MultiScaleDeformableAttention)):
                m.init_weights()
            elif isinstance(m, LQE):
                for layer in m.reg_conf.layers[:-1]:
                    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                m.init_weights()
