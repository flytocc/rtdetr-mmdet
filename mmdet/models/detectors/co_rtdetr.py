# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import Dict, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from mmcv.ops import batched_nms
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType
from mmengine.structures import InstanceData

from .deim import DEIMDFINE, DEIMRTDETR
from .dfine import DFINE
from .rtdetr import RTDETR


class CoRTDETRMixin:

    def __init__(
            self,
            *args,
            rpn_head: OptConfigType = None,  # two-stage rpn
            roi_heads: Sequence[ConfigType] = [],  # two-stage
            dense_heads: Sequence[ConfigType] = [],  # one-stage
            extra_train_cfg: Sequence[OptConfigType] = [],
            extra_test_cfg: Sequence[OptConfigType] = [],
            extra_downsample: bool = True,
            with_pos_coord: bool = True,
            max_pos_coords: int = 300,
            eval_module: Union[str, Sequence[str]] = 'detr',
            eval_roi_heads_idxs: Sequence[int] = [0],
            eval_dense_heads_idxs: Sequence[int] = [0],
            heads_nms_cfg: dict = dict(type='nms', iou_threshold=0.5),
            **kwargs) -> None:
        self.extra_downsample = extra_downsample
        self.with_pos_coord = with_pos_coord
        self.max_pos_coords = max_pos_coords
        self.eval_module = eval_module
        self.eval_roi_heads_idxs = eval_roi_heads_idxs
        self.eval_dense_heads_idxs = eval_dense_heads_idxs
        self.heads_nms_cfg = heads_nms_cfg
        co_heads = [
            head for head in roi_heads + dense_heads if 'Co' in head['type']
        ]
        self.num_co_heads = len(co_heads)
        assert len(extra_test_cfg) == len(
            extra_test_cfg) == len(roi_heads) + len(dense_heads)

        super().__init__(*args, **kwargs)

        if rpn_head is not None:
            rpn_train_cfg = extra_train_cfg[0].rpn if (extra_train_cfg[0]
                                                       is not None) else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(
                train_cfg=rpn_train_cfg, test_cfg=extra_test_cfg[0].rpn)
            rpn_head_num_classes = rpn_head_.get('num_classes', None)
            if rpn_head_num_classes is None:
                rpn_head_.update(num_classes=1)
            else:
                if rpn_head_num_classes != 1:
                    warnings.warn(
                        'The `num_classes` should be 1 in RPN, but get '
                        f'{rpn_head_num_classes}, please set '
                        'rpn_head.num_classes = 1 in your config file.')
                    rpn_head_.update(num_classes=1)
            self.rpn_head = MODELS.build(rpn_head_)
            self.proposal_cfg = extra_train_cfg[0].get('rpn_proposal',
                                                       extra_test_cfg[0].rpn)

        self.roi_heads = nn.ModuleList()
        for i, roi_head in enumerate(roi_heads):
            rcnn_train_cfg = extra_train_cfg[i].get('rcnn', None)
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=extra_test_cfg[i].rcnn)
            self.roi_heads.append(MODELS.build(roi_head))

        self.dense_heads = nn.ModuleList()
        for i, dense_head in enumerate(dense_heads, len(roi_heads)):
            dense_head.update(train_cfg=extra_train_cfg[i])
            dense_head.update(test_cfg=extra_test_cfg[i])
            self.dense_heads.append(MODELS.build(dense_head))

    def _init_layers(self):
        super()._init_layers()
        if self.num_co_heads == 0:
            return

        if self.with_pos_coord:
            self.pos_feats_trans_fc = nn.ModuleList()
            self.pos_feats_trans_norm = nn.ModuleList()
            for _ in range(self.num_co_heads):
                self.pos_feats_trans_fc.append(
                    nn.Linear(self.embed_dims, self.embed_dims))
                self.pos_feats_trans_norm.append(nn.LayerNorm(self.embed_dims))
        if self.extra_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    self.embed_dims,
                    self.embed_dims,
                    kernel_size=3,
                    stride=2,
                    padding=1), nn.BatchNorm2d(self.embed_dims))

    @property
    def query_head(self):
        return self.bbox_head

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return len(self.roi_heads) > 0

    @property
    def with_dense_head(self):
        """bool: whether the detector has a dense head"""
        return len(self.dense_heads) > 0

    def pre_transformer(
            self,
            mlvl_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        encoder_inputs_dict, decoder_inputs_dict = super().pre_transformer(
            mlvl_feats, batch_data_samples)
        self.decoder_inputs_dict = decoder_inputs_dict.copy()
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(self, mlvl_feats: Tuple[Tensor],
                        spatial_shapes: Tensor) -> Dict:
        mlvl_feats = self.encoder(mlvl_feats)
        if hasattr(self, 'downsample'):
            self.enhance_feats = (*mlvl_feats, self.downsample(mlvl_feats[-1]))
        else:
            self.enhance_feats = mlvl_feats

        feat_flatten = []
        for feat in mlvl_feats:
            batch_size, c, h, w = feat.shape
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            feat_flatten.append(feat)

        # (bs, num_feat_points, dim)
        memory = torch.cat(feat_flatten, 1)
        self.memory = memory

        encoder_outputs_dict = dict(
            memory=memory, memory_mask=None, spatial_shapes=spatial_shapes)
        return encoder_outputs_dict

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        losses = super().loss(batch_inputs, batch_data_samples)
        x = self.enhance_feats

        # RPN forward and loss
        if self.with_rpn:
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_proposal_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=self.proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        elif self.with_roi_head:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_proposal_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        positive_coords = []

        for i, roi_head in enumerate(self.roi_heads):
            roi_losses = roi_head.loss(x, rpn_proposal_list,
                                       batch_data_samples)
            if 'pos_coords' in roi_losses:
                positive_coords.append(roi_losses.pop('pos_coords'))
            losses.update(upd_loss(roi_losses, idx=i))

        for i, dense_head in enumerate(self.dense_heads, len(self.roi_heads)):
            dense_losses = dense_head.loss(x, batch_data_samples)
            if 'pos_coords' in dense_losses:
                positive_coords.append(dense_losses.pop('pos_coords'))
            losses.update(upd_loss(dense_losses, idx=i))

        if self.with_pos_coord and len(positive_coords) > 0:
            aux_dense_feats = None
            for i, (*pos_coords, head_name) in enumerate(positive_coords):
                if 'rcnn' not in head_name:
                    if aux_dense_feats is None:
                        # convert x to aux_dense_feats
                        # x: a feats_num length list of Tensor
                        # with shape (batch_size, c, w', h')
                        # aux_dense_feats: a batch_size length list of Tensor
                        # with shape (feats_num, wh', c)
                        aux_dense_feats = [
                            torch.cat([
                                feat[batch_idx].flatten(1).transpose(0, 1)
                                for feat in x
                            ]) for batch_idx in range(len(batch_inputs))
                        ]
                    pos_coords = (*pos_coords, tuple(aux_dense_feats))
                extra_losses = self.loss_aux(pos_coords, i, batch_data_samples)
                losses.update(upd_loss(extra_losses, idx=i))

        return losses

    def forward_transformer_aux(self, head_idx, aux_coords, aux_feats):
        query = self.pos_feats_trans_norm[head_idx](
            self.pos_feats_trans_fc[head_idx](aux_feats))
        decoder_outputs_dict = self.forward_decoder(
            query,
            self.memory,
            reference_points=aux_coords,
            **self.decoder_inputs_dict)
        head_inputs_dict = decoder_outputs_dict
        return head_inputs_dict

    def loss_aux(self, pos_coords, head_idx, batch_data_samples):
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)

        gt_instances, transformer_input_dict = self.query_head.get_aux_targets(
            pos_coords, self.max_pos_coords, batch_img_metas)
        head_inputs_dict = self.forward_transformer_aux(
            head_idx, **transformer_input_dict)
        losses = self.query_head.loss_aux(
            **head_inputs_dict,
            gt_instances=gt_instances,
            batch_img_metas=batch_img_metas)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        img_feats = self.extract_feat(batch_inputs)
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)
        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)
        x = self.enhance_feats

        results = {}
        if 'detr' in self.eval_module:
            tmp_dec_in, head_inputs_dict = self.pre_decoder(
                **encoder_outputs_dict, batch_data_samples=batch_data_samples)
            decoder_inputs_dict.update(tmp_dec_in)

            decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
            head_inputs_dict.update(decoder_outputs_dict)

            results_list = self.query_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)
            results['detr'] = results_list

        if 'one-stage' in self.eval_module:
            assert self.with_dense_head
            for i, dense_head in enumerate(self.dense_heads):
                if i not in self.eval_dense_heads_idxs:
                    continue
                results_list = dense_head.predict(x, batch_data_samples,
                                                  rescale)
                results[f'one-stage_{i}'] = results_list

        if 'two-stage' in self.eval_module:
            assert self.with_rpn and self.with_roi_head
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
            for i, roi_head in enumerate(self.roi_heads):
                if i not in self.eval_roi_heads_idxs:
                    continue
                results_list = roi_head.predict(x, rpn_results_list,
                                                batch_data_samples, rescale)
                results[f'two-stage_{i}'] = results_list

        assert len(results) > 0
        if len(results) > 1:
            results_list = []
            for predictions in zip(*results.values()):
                instances = InstanceData.cat(predictions)

                _, keeps = batched_nms(
                    boxes=instances.bboxes,
                    scores=instances.scores,
                    idxs=instances.labels,
                    nms_cfg=self.heads_nms_cfg)
                merged_instances = instances[keeps]
                results_list.append(merged_instances)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples


def upd_loss(losses, idx, weight=1):
    new_losses = dict()
    for k, v in losses.items():
        new_k = '{}{}'.format(k, idx)
        if isinstance(v, list) or isinstance(v, tuple):
            new_losses[new_k] = [i * weight for i in v]
        else:
            new_losses[new_k] = v * weight
    return new_losses


@MODELS.register_module()
class CoRTDETR(CoRTDETRMixin, RTDETR):
    """A CoRTDETRMixin version of RTDETR."""


@MODELS.register_module()
class CoDFINE(CoRTDETRMixin, DFINE):
    """A CoRTDETRMixin version of DFINE."""


@MODELS.register_module()
class CoDEIMDFINE(CoRTDETRMixin, DEIMDFINE):
    """A CoRTDETRMixin version of DEIMDFINE."""


@MODELS.register_module()
class CoDEIMRTDETR(CoRTDETRMixin, DEIMRTDETR):
    """A CoRTDETRMixin version of DEIMRTDETR."""
