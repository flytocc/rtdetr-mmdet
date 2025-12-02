# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import build_activation_layer
from mmengine.model import ModuleList
from torch import nn

from mmdet.registry import MODELS
from mmdet.utils import OptConfigType
from ..layers import MLP
from .dfine import DFINE, LQE
from .rtdetr import RTDETR
from .rtdetr_ins import RTDETRInsMixup


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

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        ref_hidden_dim = self.decoder.pop('ref_hidden_dim', None)
        ref_num_layers = self.decoder.pop('ref_num_layers', 2)
        ref_act_cfg = self.decoder.pop('ref_act_cfg',
                                       dict(type='SiLU', inplace=True))
        lqe_act_cfg = self.decoder.pop('lqe_act_cfg', None)

        super()._init_layers()

        # update ref_point_head
        self.decoder.ref_point_head = MLP(
            4,
            ref_hidden_dim or self.decoder.embed_dims * 2,
            self.decoder.embed_dims,
            ref_num_layers,
            act_cfg=ref_act_cfg)

        # update lqe_layers
        if lqe_act_cfg is not None:
            assert hasattr(self.decoder, 'lqe_layers')
            self.decoder.lqe_layers = ModuleList([
                LQE(4, 64, 2, self.decoder.reg_max, act_cfg=lqe_act_cfg)
                for _ in range(self.decoder.num_layers)
            ])


@MODELS.register_module()
class DEIMDFINE(DEIMMixin, DFINE):
    """DFINE for DEIM."""


@MODELS.register_module()
class DEIMRTDETR(DEIMMixin, RTDETR):
    """RTDETR for DEIM."""


@MODELS.register_module()
class DEIMDFINEIns(RTDETRInsMixup, DEIMDFINE):
    """DEIMDFINE for Instance."""
