# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional

from mmcv.cnn import Linear
from torch import nn

from mmdet.registry import MODELS
from .dfine_head import DFINEHead


@MODELS.register_module()
class DEIMV2Head(DFINEHead):
    r"""Real-Time Object Detection Meets DINOv3
    """

    def __init__(self,
                 *args,
                 share_cls_layer: Optional[bool] = None,
                 share_reg_layer: Optional[bool] = None,
                 share_pred_layer: bool = False,
                 **kwargs) -> None:
        if share_cls_layer is None:
            share_cls_layer = share_pred_layer
        if share_reg_layer is None:
            share_reg_layer = share_pred_layer
        self.share_cls_layer = share_cls_layer
        self.share_reg_layer = share_reg_layer
        super().__init__(*args, share_pred_layer=share_pred_layer, **kwargs)

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        num_wide_layers = self.num_pred_layer - self.eval_idx - 2
        scaled_dim = int(round(self.layer_scale * self.embed_dims))

        def _gen_cls_branch(embed_dims: int, out_channels: int):
            return Linear(embed_dims, out_channels)

        def _gen_reg_branch(embed_dims: int, out_channels: int):
            reg_branch = []
            for _ in range(self.num_reg_fcs):
                reg_branch.append(Linear(embed_dims, embed_dims))
                reg_branch.append(nn.ReLU())
            reg_branch.append(Linear(embed_dims, out_channels))
            return nn.Sequential(*reg_branch)

        fc_cls = _gen_cls_branch(self.embed_dims, self.cls_out_channels)
        cls_branches = [
            copy.deepcopy(fc_cls) if self.share_cls_layer else _gen_cls_branch(
                self.embed_dims, self.cls_out_channels)
            for _ in range(self.num_pred_layer - num_wide_layers - 1)
        ]
        cls_branches += [
            _gen_cls_branch(scaled_dim, self.cls_out_channels)
            for _ in range(num_wide_layers)
        ]
        cls_branches += [
            _gen_cls_branch(self.embed_dims, self.cls_out_channels)
        ]
        self.cls_branches = nn.ModuleList(cls_branches)

        reg_branch = _gen_reg_branch(self.embed_dims, 4 * (self.reg_max + 1))
        reg_branches = [
            copy.deepcopy(reg_branch) if self.share_reg_layer else
            _gen_reg_branch(self.embed_dims, 4 * (self.reg_max + 1))
            for _ in range(self.num_pred_layer - num_wide_layers - 1)
        ]
        reg_branches += [
            _gen_reg_branch(scaled_dim, 4 * (self.reg_max + 1))
            for _ in range(num_wide_layers)
        ]
        reg_branches += [_gen_reg_branch(self.embed_dims, 4)]
        reg_branches += [_gen_reg_branch(self.embed_dims, 4)]  # pre_bbox_head
        self.reg_branches = nn.ModuleList(reg_branches)
