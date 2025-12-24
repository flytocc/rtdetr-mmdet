# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmdet.models.layers.transformer import inverse_sigmoid
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from mmdet.utils import ConfigType, OptConfigType
from ..layers import DnQueryGenerator
from .rtdetr import RTDETR


class RTDETRInsMixup:
    """Mixup for RTDETR family Instance Model."""

    def __init__(self,
                 *args,
                 mask_feat_cfg: ConfigType = dict(
                     in_channels=256,
                     feat_channels=128,
                     num_prototypes=256,
                     act_cfg=dict(type='ReLU', inplace=True),
                     norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
                 **kwargs) -> None:
        self.mask_feat_cfg = mask_feat_cfg
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        super()._init_layers()
        self.mask_features = MaskFeatModule_ppdet(**self.mask_feat_cfg)
        self.decoder.norm = nn.LayerNorm(self.embed_dims)

    def forward_encoder(self, mlvl_feats: Tuple[Tensor],
                        spatial_shapes: Tensor) -> Dict:
        """Forward with Transformer encoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            mlvl_feats (tuple[Tensor]): Multi-level features that may have
                different resolutions, output from neck. Each feature has
                shape (bs, dim, h_lvl, w_lvl), where 'lvl' means 'layer'.
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).

        Returns:
            dict: The output of the Transformer encoder, which includes
            `memory`,  `mask_features` and `spatial_shapes`.
        """
        mlvl_feats = self.encoder(mlvl_feats)

        # for mask
        mask_features = self.mask_features(mlvl_feats)

        feat_flatten = []
        for feat in mlvl_feats:
            batch_size, c, h, w = feat.shape
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            feat_flatten.append(feat)

        # (bs, num_feat_points, dim)
        memory = torch.cat(feat_flatten, 1)

        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=None,
            mask_features=mask_features,
            spatial_shapes=spatial_shapes)
        return encoder_outputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        mask_features: Tensor,
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points). Will only be used when
                `as_two_stage` is `True`.
            mask_features (Tensor): instance mask features that
                has shape (bs, dim, h, w).
            spatial_shapes (Tensor): Spatial shapes of features in all levels.
                With shape (num_levels, 2), last dimension represents (h, w).
                Will only be used when `as_two_stage` is `True`.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The decoder_inputs_dict and head_inputs_dict.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'memory',
              `reference_points`, and `dn_mask`. The reference points of
              decoder input here are 4D boxes, although it has `points`
              in its name.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which includes `topk_score`, `topk_coords`,
              `mask_features` and `dn_meta` when `self.training` is `True`,
              else is empty.
        """
        bs, _, c = memory.shape
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].out_features

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](
                self.decoder.norm(output_memory))

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        query = torch.gather(output_memory, 1,
                             topk_indices.unsqueeze(-1).repeat(1, 1, c))

        # for mask
        enc_mask_feat = self.bbox_head.mask_branches[
            self.decoder.num_layers](self.decoder.norm(query))
        topk_mask = self.bbox_head.feat_to_mask(enc_mask_feat,
                                                mask_features)

        # unified reference points
        h, w = topk_mask.shape[-2:]
        factor = topk_mask.new_tensor([w, h, w, h]).unsqueeze(0)
        # mask to box is a non-differentiable operation
        masks = topk_mask.detach().reshape(-1, h, w) > 0
        topk_coords_xyxy = mask2bbox_onnx_export(masks).reshape(bs, -1, 4)
        topk_coords_normalized = bbox_xyxy_to_cxcywh(topk_coords_xyxy) / factor
        topk_coords_unact = inverse_sigmoid(topk_coords_normalized)

        if self.training:
            topk_score = torch.gather(
                enc_outputs_class, 1,
                topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
            topk_output_proposals = torch.gather(
                output_proposals, 1,
                topk_indices.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact_ori = self.bbox_head.reg_branches[
                self.decoder.num_layers](query) + topk_output_proposals
            topk_coords = topk_coords_unact_ori.sigmoid()

            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = query.detach()  # detach() is not used in DINO
            query = torch.cat([dn_label_query, query], dim=1)
            dn_bbox_query = dn_bbox_query.type_as(topk_coords_unact)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        # NOTE To avoid inverse_sigmoid in decoder
        # reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            cls_branches=self.bbox_head.cls_branches,
            eval_idx=self.eval_idx)
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            enc_outputs_mask=topk_mask,
            mask_features=mask_features,
            dn_meta=dn_meta) if self.training else dict(
                mask_features=mask_features)
        return decoder_inputs_dict, head_inputs_dict


@MODELS.register_module()
class RTDETRIns(RTDETRInsMixup, RTDETR):
    """RTDETR for Instance."""


class RTDETRInsPlusMixup:

    def __init__(self,
                 *args,
                 num_prototypes: int = 64,
                 mask_dims: int = 32,
                 **kwargs) -> None:
        self.num_prototypes = num_prototypes
        self.mask_dims = mask_dims
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        super()._init_layers()
        self.enc_mask_output = nn.Sequential(
            ConvModule(
                self.num_prototypes,
                self.num_prototypes,
                3,
                padding=1,
                act_cfg=dict(type='SiLU', inplace=True),
                norm_cfg=dict(type='BN')),
            ConvModule(
                self.num_prototypes,
                self.mask_dims,
                1,
                act_cfg=None)
        ) if self.num_prototypes != self.mask_dims else nn.Identity()

    def pre_transformer(
            self,
            mlvl_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        c2_feat, *mlvl_feats = mlvl_feats
        encoder_inputs_dict, decoder_inputs_dict = super().pre_transformer(
            mlvl_feats, batch_data_samples)
        encoder_inputs_dict['c2_feat'] = c2_feat
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(self, c2_feat: Tensor, mlvl_feats: Tuple[Tensor],
                        spatial_shapes: Tensor) -> Dict:
        encoder_outputs_dict = super().forward_encoder(mlvl_feats,
                                                       spatial_shapes)
        mask_features = encoder_outputs_dict.pop('mask_features')
        mask_features = c2_feat + F.interpolate(
            mask_features, size=c2_feat.shape[-2:], mode='bilinear')
        encoder_outputs_dict['mask_features'] = self.enc_mask_output(
            mask_features)
        return encoder_outputs_dict


@MODELS.register_module()
class RTDETRInsPlus(RTDETRInsPlusMixup, RTDETRIns):
    """RTDETRInsPlus with C2"""


def mask2bbox_onnx_export(masks: Tensor) -> Tensor:
    N, H, W = masks.shape

    x_any = torch.any(masks, dim=2)  # (N, H)
    y_any = torch.any(masks, dim=1)  # (N, W)
    x_sum = x_any.int()
    y_sum = y_any.int()

    xmin = torch.argmax(y_sum, dim=1)  # (N,)
    ymin = torch.argmax(x_sum, dim=1)  # (N,)
    xmax_rev = torch.argmax(torch.flip(y_sum, dims=[1]), dim=1)  # (N,)
    ymax_rev = torch.argmax(torch.flip(x_sum, dims=[1]), dim=1)  # (N,)
    xmax = W - xmax_rev
    ymax = H - ymax_rev

    bboxes = torch.stack([xmin, ymin, xmax, ymax], dim=1)  # (N, 4)

    is_not_empty = torch.any(y_any, dim=1).unsqueeze(-1)  # (N, 1)
    bboxes *= is_not_empty.int()
    return bboxes.float()


class MaskFeatModule_ppdet(BaseModule):

    def __init__(
        self,
        in_channels: int,
        feat_channels: int = 256,
        num_prototypes: int = 8,
        act_cfg: ConfigType = dict(type='SiLU', inplace=True),
        norm_cfg: ConfigType = dict(type='BN')
    ) -> None:
        super().__init__(init_cfg=None)

        fpn_strides = [8, 16, 32]
        if isinstance(in_channels, int):
            in_channels = [in_channels] * len(fpn_strides)
        assert len(in_channels) == len(fpn_strides)
        reorder_index = np.argsort(fpn_strides, axis=0)
        in_channels = [in_channels[i] for i in reorder_index]
        fpn_strides = [fpn_strides[i] for i in reorder_index]
        assert min(fpn_strides) == fpn_strides[0]
        self.reorder_index = reorder_index
        self.fpn_strides = fpn_strides

        self.scale_heads = nn.ModuleList()
        for i in range(len(fpn_strides)):
            head_length = max(
                1, int(np.log2(fpn_strides[i]) - np.log2(fpn_strides[0])))
            scale_head = []
            for k in range(head_length):
                in_c = in_channels[i] if k == 0 else feat_channels
                scale_head.append(
                    ConvModule(
                        in_c,
                        feat_channels,
                        3,
                        padding=1,
                        act_cfg=act_cfg,
                        norm_cfg=norm_cfg))
                if fpn_strides[i] != fpn_strides[0]:
                    scale_head.append(
                        nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=False))
            self.scale_heads.append(nn.Sequential(*scale_head))

        self.output_conv = ConvModule(
            feat_channels,
            num_prototypes,
            3,
            padding=1,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg)

    def forward(self, inputs):
        x = [inputs[i] for i in self.reorder_index]
        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.fpn_strides)):
            output = output + self.scale_heads[i](x[i])
        output = self.output_conv(output)
        return output


@MODELS.register_module()
class MaskRTDETR_ppdet(RTDETRInsPlusMixup, RTDETRIns):
    """MaskRTDETR in PaddleDetection

    Args:
        dn_cfg (:obj:`ConfigDict` or dict, optional): Config of denoising
            query generator. Defaults to `None`.
    """

    def __init__(self, *args, dn_cfg: OptConfigType = None, **kwargs) -> None:
        super().__init__(*args, dn_cfg=dn_cfg, **kwargs)
        self.dn_query_generator = DnQueryGenerator(**dn_cfg)

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        mask_features: Tensor,
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points). Will only be used when
                `as_two_stage` is `True`.
            mask_features (Tensor): instance mask features that
                has shape (bs, dim, h, w).
            spatial_shapes (Tensor): Spatial shapes of features in all levels.
                With shape (num_levels, 2), last dimension represents (h, w).
                Will only be used when `as_two_stage` is `True`.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The decoder_inputs_dict and head_inputs_dict.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'memory',
              `reference_points`, and `dn_mask`. The reference points of
              decoder input here are 4D boxes, although it has `points`
              in its name.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which includes `topk_score`, `topk_coords`,
              `mask_features` and `dn_meta` when `self.training` is `True`,
              else is empty.
        """
        decoder_inputs_dict, head_inputs_dict = super().pre_decoder(
            memory, memory_mask, mask_features, spatial_shapes,
            batch_data_samples)

        # NOTE loss for init_outputs may be bad for training
        if self.training:
            init_coords = decoder_inputs_dict['reference_points'].sigmoid()
            norm_query = self.decoder.norm(decoder_inputs_dict['query'])
            init_score = self.bbox_head.cls_branches[
                self.decoder.num_layers](norm_query)
            init_mask_feat = self.bbox_head.mask_branches[
                self.decoder.num_layers](norm_query)
            init_mask = self.bbox_head.feat_to_mask(init_mask_feat,
                                                    mask_features)
            head_inputs_dict['init_outputs_class'] = init_score
            head_inputs_dict['init_outputs_coord'] = init_coords
            head_inputs_dict['init_outputs_mask'] = init_mask

        return decoder_inputs_dict, head_inputs_dict
