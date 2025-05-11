# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import build_activation_layer
from torch import nn

from mmdet.registry import MODELS
from mmdet.utils import OptConfigType
from ..layers import MLP
from .dfine import DFINE
from .rtdetr import RTDETR


class DEIMMixin:

    def __init__(self,
                 *args,
                 bbox_head: OptConfigType = None,
                 **kwargs) -> None:
        reg_act_cfg = bbox_head.pop(
            'reg_act_cfg', dict(type='SiLU', inplace=True))
        super().__init__(*args, bbox_head=bbox_head, **kwargs)
        for reg_branche in self.bbox_head.reg_branches:
            for idx, layer in enumerate(reg_branche):
                if isinstance(layer, nn.ReLU):
                    reg_branche[idx] = build_activation_layer(reg_act_cfg)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        ref_act_cfg = self.decoder.pop(
            'ref_act_cfg', dict(type='SiLU', inplace=True))
        ref_hidden_dim = self.decoder.pop('ref_hidden_dim', None)
        ref_num_layers = self.decoder.pop('ref_num_layers', 2)

        super()._init_layers()

        self.decoder.ref_point_head = MLP(
            4,
            ref_hidden_dim or self.decoder.embed_dims * 2,
            self.decoder.embed_dims,
            ref_num_layers,
            act_cfg=ref_act_cfg)


@MODELS.register_module()
class DEIMDFINE(DEIMMixin, DFINE):
    """DFINE for DEIM."""


@MODELS.register_module()
class DEIMRTDETR(DEIMMixin, RTDETR):
    """RTDETR for DEIM."""
