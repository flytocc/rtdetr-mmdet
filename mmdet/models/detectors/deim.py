# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import build_activation_layer
from torch import nn

from mmdet.registry import MODELS
from mmdet.utils import OptConfigType
from .dfine import DFINE
from .rtdetr import RTDETR


class DEIMMixin:
    r"""Implementation of `DEIM: DETR with Improved Matching for Fast
    Convergence <https://arxiv.org/abs/2412.04234>`_

    Code is modified from the `official github repo
    <https://github.com/ShihuaHuang95/DEIM>`_.

    Args:
        bbox_head (:obj:`ConfigDict` or dict, optional): Config of bbox head.
            Defaults to `None`.
    """

    def __init__(self,
                 *args,
                 bbox_head: OptConfigType = None,
                 **kwargs) -> None:
        reg_act_cfg = bbox_head.pop('reg_act_cfg',
                                    dict(type='SiLU', inplace=True))
        super().__init__(*args, bbox_head=bbox_head, **kwargs)
        for reg_branche in self.bbox_head.reg_branches:
            for idx, layer in enumerate(reg_branche):
                if isinstance(layer, nn.ReLU):
                    reg_branche[idx] = build_activation_layer(reg_act_cfg)


@MODELS.register_module()
class DEIMDFINE(DEIMMixin, DFINE):
    """DFINE for DEIM."""


@MODELS.register_module()
class DEIMRTDETR(DEIMMixin, RTDETR):
    """RTDETR for DEIM."""
