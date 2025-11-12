# Copyright (c) OpenMMLab. All rights reserved.
import math

from mmengine.logging import MMLogger
from torch import nn

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.utils import ConfigType
from ..layers import MLP, DEIMV2TransformerDecoder, RTDETRHybridEncoder
from ..layers.transformer.dfine_layers import (
    LQE, Gate, MultiNumPointsMultiScaleDeformableAttention)
from .deformable_detr import DeformableDETR, MultiScaleDeformableAttention
from .deim import DEIMDFINE


@MODELS.register_module()
class DEIMV2(DEIMDFINE):
    """Implementation of `Real-Time Object Detection Meets DINOv3.

    <https://arxiv.org/abs/2509.20787>`_
    """

    def __init__(self,
                 *args,
                 train_cfg: ConfigType = dict(
                     assigner=dict(
                         type='HungarianAssigner',
                         match_costs=[
                             dict(type='ClassificationCost', weight=1.),
                             dict(
                                 type='BBoxL1Cost',
                                 weight=5.0,
                                 box_format='xywh'),
                             dict(type='IoUCost', iou_mode='giou', weight=2.0)
                         ]),
                     switch_assigner=dict(
                         switch_epoch=45,
                         assigner=dict(
                             type='HungarianAssigner',
                             match_costs=[
                                 dict(
                                     type='DEIMV2LossCost',
                                     iou_order_alpha=4.0,
                                     weight=1.)
                             ]))),
                 **kwargs) -> None:
        super().__init__(*args, train_cfg=train_cfg, **kwargs)

        if train_cfg and 'switch_assigner' in train_cfg:
            switch_assigner_cfg = train_cfg['switch_assigner']
            self.switch_assigner_epoch = switch_assigner_cfg['switch_epoch']
            self.switch_assigner = TASK_UTILS.build(
                switch_assigner_cfg['assigner'])
            self.assigner_has_switched = False

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.encoder = RTDETRHybridEncoder(**self.encoder)
        self.decoder = DEIMV2TransformerDecoder(**self.decoder)
        self.embed_dims = self.decoder.embed_dims
        self.memory_trans_fc = nn.Identity()
        self.memory_trans_norm = nn.Identity()

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

    def _switch_assigner(self) -> None:
        """Switch to the new assigner during training."""
        if hasattr(self, 'switch_assigner_epoch'):
            if not hasattr(self, 'epoch'):
                raise AttributeError(
                    'Please set the current epoch number to the model '
                    'before calling loss function. Use `SetEpochInfoHook`')
            epoch_to_be_switched = self.epoch >= self.switch_assigner_epoch
            if epoch_to_be_switched and not self.assigner_has_switched:
                logger = MMLogger.get_current_instance()
                logger.info('Switching to the new assigner at epoch '
                            f'{self.epoch}.')
                assert hasattr(self.bbox_head, 'assigner'), \
                    'The bbox_head must have an assigner to be switched.'
                self.bbox_head.assigner = self.switch_assigner
                self.assigner_has_switched = True

    def set_epoch(self, value: int) -> None:
        """Set current epoch number and switch assigner if needed.

        Note:
            This function is called by `SetEpochInfoHook` during training.
        """
        self.epoch = value
        self._switch_assigner()
