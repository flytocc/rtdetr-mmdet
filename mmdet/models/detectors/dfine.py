# Copyright (c) OpenMMLab. All rights reserved.
import math

from torch import nn

from mmdet.registry import MODELS
from mmdet.utils import OptConfigType
from ..layers import (DFINECdnQueryGenerator, DFINETransformerDecoder,
                      RTDETRHybridEncoder)
from ..layers.transformer.dfine_layers import (
    LQE, Gate, MultiNumPointsMultiScaleDeformableAttention)
from .rtdetr import RTDETR


@MODELS.register_module()
class DFINE(RTDETR):
    r"""Implementation of `D-FINE: REDEFINE REGRESSION TASK IN DETRS AS
    FINE-GRAINED DISTRIBUTION REFINEMENT <https://arxiv.org/abs/2410.13842>`_

    Code is modified from the `official github repo
    <https://github.com/Peterande/D-FINE>`_.

    Args:
        dn_cfg (:obj:`ConfigDict` or dict, optional): Config of denoising
            query generator. Defaults to `None`.
    """

    def __init__(self, *args, dn_cfg: OptConfigType = None, **kwargs) -> None:
        super().__init__(*args, dn_cfg=dn_cfg, **kwargs)
        self.dn_query_generator = DFINECdnQueryGenerator(**dn_cfg)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.encoder = RTDETRHybridEncoder(**self.encoder)
        self.decoder = DFINETransformerDecoder(**self.decoder)
        self.embed_dims = self.decoder.embed_dims
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        for m in self.modules():
            if isinstance(m,
                          (Gate, MultiNumPointsMultiScaleDeformableAttention)):
                m.init_weights()
            elif isinstance(m, LQE):
                for layer in m.reg_conf.layers[:-1]:
                    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                m.init_weights()
