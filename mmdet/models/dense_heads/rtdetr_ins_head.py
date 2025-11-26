# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmcv.ops import batched_nms, point_sample
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import (bbox_cxcywh_to_xyxy, bbox_overlaps,
                                   bbox_xyxy_to_cxcywh)
from mmdet.structures.mask import BaseInstanceMasks
from mmdet.utils import ConfigType, InstanceList, OptInstanceList, reduce_mean
from ..losses import RTDETRVarifocalLoss
from ..utils import get_uncertain_point_coords_with_randomness, multi_apply
from .rtdetr_head import RTDETRHead


class RTDETRInsHeadMixup:
    """
    Args:
        mask_dims (int): The dims of mask head embedding.
        num_mask_fcs (int): Number of fully-connected layers used in `FFN`,
            which is then used for the mask head. Defaults to 3.
        share_mask_layer (bool)
        loss_mask (:obj:`ConfigDict` or dict): Config of the mask loss.
            Defaults to None.
        loss_dice (:obj:`ConfigDict` or dict): Config of the dice loss.
            Defaults to None.
    """

    def __init__(self,
                 *args,
                 embed_dims: int = 256,
                 mask_dims: Optional[int] = None,
                 num_mask_fcs: int = 3,
                 share_mask_layer: Optional[bool] = True,
                 share_pred_layer: bool = False,
                 vfl_iou_type: Literal['bbox', 'mask'] = 'bbox',
                 loss_mask: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=5.0),
                 loss_dice: ConfigType = dict(
                     type='DiceLoss',
                     use_sigmoid=True,
                     activate=True,
                     reduction='mean',
                     naive_dice=True,
                     eps=1.0,
                     loss_weight=5.0),
                 **kwargs) -> None:
        self.mask_dims = mask_dims or embed_dims
        self.num_mask_fcs = num_mask_fcs
        if share_mask_layer is None:
            share_mask_layer = share_pred_layer
        self.share_mask_layer = share_mask_layer
        self.vfl_iou_type = vfl_iou_type
        super().__init__(*args, embed_dims=embed_dims,
                         share_pred_layer=share_pred_layer, **kwargs)

        if self.train_cfg:
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.loss_mask = MODELS.build(loss_mask)
        self.loss_dice = MODELS.build(loss_dice)

    def feat_to_mask(self, mask_preds: Tensor, mask_feats: Tensor) -> Tensor:
        b, _, h, w = mask_feats.shape
        return (mask_preds @ mask_feats.flatten(-2)).view(b, -1, h, w)

    def forward(self, hidden_states: Tensor, references: List[Tensor],
                mask_features: Tensor) -> Tuple[Tensor, Tensor]:
        outs = super().forward(hidden_states, references)

        all_layers_outputs_masks = []

        for layer_id, hidden_state in hidden_states:
            tmp_mask_preds = self.mask_branches[layer_id](hidden_state)
            outputs_mask = self.feat_to_mask(tmp_mask_preds, mask_features)
            all_layers_outputs_masks.append(outputs_mask)

        return (*outs, all_layers_outputs_masks)

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             enc_outputs_mask: Tensor, mask_features: Tensor,
             batch_data_samples: SampleList, dn_meta: Dict[str, int]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries_total,
                dim), where `num_queries_total` is the sum of
                `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries_total, 4) and each `inter_reference` has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            enc_outputs_class (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_outputs_coord (Tensor): The proposal generate from the
                encode feature map, has shape (bs, num_feat_points, 4) with the
                last dimension arranged as (cx, cy, w, h).
            enc_outputs_mask (Tensor): The mask generate from the
                encode feature map, has shape (bs, num_feat_points,
                mask_out_channels).
            mask_features (Tensor): instance mask features that
                has shape (bs, dim, h, w).
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            gt_instances = data_sample.gt_instances
            if isinstance(gt_instances.masks, BaseInstanceMasks):
                gt_instances.masks = gt_instances.masks.to_tensor(
                    dtype=torch.bool, device=mask_features.device)
            assert isinstance(gt_instances.masks, Tensor)
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references, mask_features)
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              enc_outputs_mask, batch_gt_instances,
                              batch_img_metas, dn_meta)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                mask_features: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, num_queries, bs, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Defaults to `True`.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self(hidden_states, references, mask_features)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions

    def predict_by_feat(self,
                        all_layers_cls_scores: Tensor,
                        all_layers_bbox_preds: Tensor,
                        all_layers_mask_preds: Tensor,
                        batch_img_metas: List[Dict],
                        rescale: bool = False) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs, num_queries,
                cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and shape (num_decoder_layers, bs, num_queries,
                4) with the last dimension arranged as (cx, cy, w, h).
            batch_img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Default `False`.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        cls_scores = all_layers_cls_scores[-1]
        bbox_preds = all_layers_bbox_preds[-1]
        mask_preds = all_layers_mask_preds[-1]

        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            mask_pred = mask_preds[img_id]
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(cls_score, bbox_pred,
                                                   mask_pred, img_meta,
                                                   rescale)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                mask_pred: Tensor,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_queries, 4].
            img_meta (dict): Image meta info.
            rescale (bool): If True, return boxes in original image
                space. Default True.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_score) == len(bbox_pred)  # num_queries
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        max_per_img = min(max_per_img, len(cls_score))
        img_shape = img_meta['img_shape']
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
            mask_pred = mask_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            mask_pred = mask_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])

        mask_pred = F.interpolate(
            mask_pred.unsqueeze(1),
            size=img_shape,
            mode='bilinear',
            align_corners=False)

        if rescale:
            assert img_meta.get('scale_factor') is not None
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))
            ori_h, ori_w = img_meta['ori_shape'][:2]
            mask_pred = F.interpolate(
                mask_pred,
                size=[
                    math.ceil(img_shape[0] / img_meta['scale_factor'][1]),
                    math.ceil(img_shape[1] / img_meta['scale_factor'][0])
                ],
                mode='bilinear',
                align_corners=False)[..., :ori_h, :ori_w]

        masks = mask_pred.squeeze(1)
        masks = masks.sigmoid() > self.test_cfg.mask_thr_binary

        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels
        results.masks = masks

        score_thr = self.test_cfg.get('score_thr', -1)
        if score_thr > 0:
            valid_mask = torch.gt(results.scores, score_thr)
            if not valid_mask.all():
                results = results[valid_mask]

        nms_cfg = self.test_cfg.get('nms', None)
        if nms_cfg is not None:
            _, keeps = batched_nms(
                boxes=results.bboxes,
                scores=results.scores,
                idxs=results.labels,
                nms_cfg=nms_cfg)
            if not keeps.all():
                results = results[keeps]

        return results


@MODELS.register_module()
class RTDETRInsHead(RTDETRInsHeadMixup, RTDETRHead):
    """RTDETR Head for Instance.
    """

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        super()._init_layers()
        mask_branch = []
        for _ in range(self.num_mask_fcs):
            mask_branch.append(Linear(self.embed_dims, self.embed_dims))
            mask_branch.append(nn.ReLU())
        mask_branch.append(Linear(self.embed_dims, self.mask_dims))
        mask_branch = nn.Sequential(*mask_branch)

        if self.share_mask_layer:
            self.mask_branches = nn.ModuleList(
                [mask_branch for _ in range(self.num_pred_layer)])
        else:
            self.mask_branches = nn.ModuleList([
                copy.deepcopy(mask_branch) for _ in range(self.num_pred_layer)
            ])

    @staticmethod
    def split_outputs(all_layers_cls_scores: Tensor,
                      all_layers_bbox_preds: Tensor,
                      all_layers_mask_preds: Tensor,
                      dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Split outputs of the denoising part and the matching part.

        For the total outputs of `num_queries_total` length, the former
        `num_denoising_queries` outputs are from denoising queries, and
        the rest `num_matching_queries` ones are from matching queries,
        where `num_queries_total` is the sum of `num_denoising_queries` and
        `num_matching_queries`.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'.

        Returns:
            Tuple[Tensor]: a tuple containing the following outputs.

            - all_layers_matching_cls_scores (Tensor): Classification scores
              of all decoder layers in matching part, has shape
              (num_decoder_layers, bs, num_matching_queries, cls_out_channels).
            - all_layers_matching_bbox_preds (Tensor): Regression outputs of
              all decoder layers in matching part. Each is a 4D-tensor with
              normalized coordinate format (cx, cy, w, h) and has shape
              (num_decoder_layers, bs, num_matching_queries, 4).
            - all_layers_denoising_cls_scores (Tensor): Classification scores
              of all decoder layers in denoising part, has shape
              (num_decoder_layers, bs, num_denoising_queries,
              cls_out_channels).
            - all_layers_denoising_bbox_preds (Tensor): Regression outputs of
              all decoder layers in denoising part. Each is a 4D-tensor with
              normalized coordinate format (cx, cy, w, h) and has shape
              (num_decoder_layers, bs, num_denoising_queries, 4).
        """
        if dn_meta is not None:
            num_denoising_queries = dn_meta['num_denoising_queries']
            all_layers_denoising_mask_preds = \
                [o[:, : num_denoising_queries] for o in all_layers_mask_preds]
            all_layers_matching_mask_preds = \
                [o[:, num_denoising_queries:] for o in all_layers_mask_preds]
        else:
            all_layers_denoising_mask_preds = None
            all_layers_matching_mask_preds = all_layers_mask_preds

        (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
         all_layers_denoising_cls_scores,
         all_layers_denoising_bbox_preds) = RTDETRHead.split_outputs(
             all_layers_cls_scores, all_layers_bbox_preds, dn_meta)

        return (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
                all_layers_matching_mask_preds,
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
                all_layers_denoising_mask_preds)

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        all_layers_mask_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        enc_mask_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels), where
                `num_queries_total` is the sum of `num_denoising_queries`
                and `num_matching_queries`.
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            enc_cls_scores (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_bbox_preds (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
                group collation, including 'num_denoising_queries' and
                'num_denoising_groups'. It will be used for split outputs of
                denoising and matching parts and loss calculation.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # extract denoising and matching part of outputs
        (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
         all_layers_matching_mask_preds, all_layers_denoising_cls_scores,
         all_layers_denoising_bbox_preds, all_layers_denoising_mask_preds) = \
            self.split_outputs(
                all_layers_cls_scores, all_layers_bbox_preds,
                all_layers_mask_preds, dn_meta)

        (losses_cls, losses_bbox, losses_iou, losses_mask,
         losses_dice) = multi_apply(
             self.loss_by_feat_single,
             all_layers_matching_cls_scores,
             all_layers_matching_bbox_preds,
             all_layers_matching_mask_preds,
             batch_gt_instances=batch_gt_instances,
             batch_img_metas=batch_img_metas)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_mask_i, loss_dice_i in \
                zip(losses_cls[:-1], losses_bbox[:-1],
                    losses_iou[:-1], losses_mask[:-1], losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            num_dec_layer += 1

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            # NOTE The enc_loss calculation of the DINO is
            # different from that of Deformable DETR.
            (enc_loss_cls, enc_loss_bbox, enc_loss_iou, enc_loss_mask,
             enc_loss_dice) = \
                self.loss_by_feat_single(
                    enc_cls_scores, enc_bbox_preds, enc_mask_preds,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_loss_bbox
            loss_dict['enc_loss_iou'] = enc_loss_iou
            loss_dict['enc_loss_mask'] = enc_loss_mask
            loss_dict['enc_loss_dice'] = enc_loss_dice

        if all_layers_denoising_cls_scores is not None:
            # calculate denoising loss from all decoder layers
            (dn_losses_cls, dn_losses_bbox, dn_losses_iou, dn_losses_mask,
             dn_losses_dice) = self.loss_dn(
                 all_layers_denoising_cls_scores,
                 all_layers_denoising_bbox_preds,
                 all_layers_denoising_mask_preds,
                 batch_gt_instances=batch_gt_instances,
                 batch_img_metas=batch_img_metas,
                 dn_meta=dn_meta)
            # collate denoising loss
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            loss_dict['dn_loss_mask'] = dn_losses_mask[-1]
            loss_dict['dn_loss_dice'] = dn_losses_dice[-1]
            for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i,
                                loss_mask_i, loss_dice_i) in \
                    enumerate(zip(dn_losses_cls[:-1], dn_losses_bbox[:-1],
                                  dn_losses_iou[:-1], dn_losses_mask[:-1],
                                  dn_losses_dice[:-1])):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
                loss_dict[f'd{num_dec_layer}.dn_loss_mask'] = loss_mask_i
                loss_dict[f'd{num_dec_layer}.dn_loss_dice'] = loss_dice_i

        return loss_dict

    def loss_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor,
                            mask_preds: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images, has shape (bs, num_queries, cls_out_channels).
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape (bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           mask_preds_list, batch_gt_instances,
                                           batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mask_targets_list, mask_weights_list, num_total_pos,
         num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        mask_targets = torch.cat(mask_targets_list, 0)
        mask_weights = torch.stack(mask_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor])).item()
        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, RTDETRVarifocalLoss):
            cls_iou_targets = label_weights.new_zeros(cls_scores.shape)
            bg_class_ind = self.num_classes
            pos_inds = ((labels >= 0)
                        & (labels < bg_class_ind)).nonzero().squeeze(1)
            pos_labels = labels[pos_inds]
            if self.vfl_iou_type == 'mask':
                pos_mask_preds = mask_preds.detach()[mask_weights > 0]
                pos_mask_preds = F.interpolate(
                    pos_mask_preds.unsqueeze(1),
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=False).squeeze(1)
                pos_mask_preds = (
                    pos_mask_preds > self.test_cfg.mask_thr_binary).float()

                pos_mask_targets = mask_targets.float()
                if pos_mask_preds.shape[-1] != pos_mask_targets.shape[-1] \
                        or pos_mask_preds.shape[-2] != pos_mask_targets.shape[-2]:
                    pos_mask_targets = F.interpolate(
                        pos_mask_targets.unsqueeze(1),
                        size=pos_mask_preds.shape[-2:],
                        mode='bilinear',
                        align_corners=False).squeeze(1)

                cls_iou_targets[pos_inds, pos_labels] = mask_overlaps(
                    pos_mask_preds, pos_mask_targets)
            else:
                pos_bbox_targets = bbox_targets[pos_inds]
                pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
                pos_bbox_pred = bbox_preds.detach().reshape(-1, 4)[pos_inds]
                pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
                cls_iou_targets[pos_inds, pos_labels] = bbox_overlaps(
                    pos_decode_bbox_pred,
                    pos_decode_bbox_targets,
                    is_aligned=True)

            loss_cls = self.loss_cls(
                cls_scores, cls_iou_targets, avg_factor=cls_avg_factor)
        else:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        if self.bg_cls_weight == 0:
            num_total_pos = cls_avg_factor
        else:
            num_total_pos = bbox_preds.new_tensor([num_total_pos])
            num_total_pos = torch.clamp(
                reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_pos)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_pos * self.num_points)

        return loss_cls, loss_bbox, loss_iou, loss_mask, loss_dice

    @torch.no_grad()
    def get_targets(self, cls_scores_list: List[Tensor],
                    bbox_preds_list: List[Tensor],
                    mask_preds_list: List[Tensor],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict]) -> Tuple:
        """Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image, has shape [num_queries,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_queries, 4].
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple: a tuple containing the following targets.

            - labels_list (list[Tensor]): Labels for all images.
            - label_weights_list (list[Tensor]): Label weights for all images.
            - bbox_targets_list (list[Tensor]): BBox targets for all images.
            - bbox_weights_list (list[Tensor]): BBox weights for all images.
            - num_total_pos (int): Number of positive samples in all images.
            - num_total_neg (int): Number of negative samples in all images.
        """
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mask_targets_list, mask_weights_list, pos_inds_list,
         neg_inds_list) = multi_apply(self._get_targets_single,
                                      cls_scores_list, bbox_preds_list,
                                      mask_preds_list, batch_gt_instances,
                                      batch_img_metas)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, mask_targets_list, mask_weights_list,
                num_total_pos, num_total_neg)

    def _get_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            mask_pred: Tensor, gt_instances: InstanceData,
                            img_meta: dict) -> Tuple:
        """Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_queries, 4].
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_masks = gt_instances.masks

        img_h, img_w = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        num_bboxes = bbox_pred.size(0)
        # convert bbox_pred from xywh, normalized to xyxy, unnormalized
        bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_pred = bbox_pred * factor

        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)
        # shape (num_queries, num_points)
        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1,
                                                        1)).squeeze(1)
        # shape (num_gts, num_points)
        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1,
                                                               1)).squeeze(1)

        pred_instances = InstanceData(
            scores=cls_score, bboxes=bbox_pred, masks=mask_points_pred)
        gt_instances = InstanceData(
            labels=gt_labels, bboxes=gt_bboxes, masks=gt_points_masks)
        # assigner and sampler
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)

        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights[pos_inds] = 1.0

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        # mask targets
        mask_targets = gt_masks[pos_assigned_gt_inds.long(), :, :]
        mask_weights = mask_pred.new_zeros(num_bboxes)
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, bbox_targets, bbox_weights,
                mask_targets, mask_weights, pos_inds, neg_inds)

    def loss_dn(self, all_layers_denoising_cls_scores: Tensor,
                all_layers_denoising_bbox_preds: Tensor,
                all_layers_denoising_mask_preds: Tensor,
                batch_gt_instances: InstanceList, batch_img_metas: List[dict],
                dn_meta: Dict[str, int]) -> Tuple[List[Tensor]]:
        """Calculate denoising loss.

        Args:
            all_layers_denoising_cls_scores (Tensor): Classification scores of
                all decoder layers in denoising part, has shape (
                num_decoder_layers, bs, num_denoising_queries,
                cls_out_channels).
            all_layers_denoising_bbox_preds (Tensor): Regression outputs of all
                decoder layers in denoising part. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and has shape
                (num_decoder_layers, bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[List[Tensor]]: The loss_dn_cls, loss_dn_bbox, and loss_dn_iou
            of each decoder layers.
        """
        return multi_apply(
            self._loss_dn_single,
            all_layers_denoising_cls_scores,
            all_layers_denoising_bbox_preds,
            all_layers_denoising_mask_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            dn_meta=dn_meta)

    def _loss_dn_single(self, dn_cls_scores: Tensor, dn_bbox_preds: Tensor,
                        dn_mask_preds: Tensor,
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Denoising loss for outputs from a single decoder layer.

        Args:
            dn_cls_scores (Tensor): Classification scores of a single decoder
                layer in denoising part, has shape (bs, num_denoising_queries,
                cls_out_channels).
            dn_bbox_preds (Tensor): Regression outputs of a single decoder
                layer in denoising part. Each is a 4D-tensor with normalized
                coordinate format (cx, cy, w, h) and has shape
                (bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        cls_reg_targets = self.get_dn_targets(batch_gt_instances,
                                              batch_img_metas, dn_meta)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mask_targets_list, mask_weights_list, num_total_pos,
         num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        mask_targets = torch.cat(mask_targets_list, 0)
        mask_weights = torch.stack(mask_weights_list, 0)

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor])).item()
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            if isinstance(self.loss_cls, RTDETRVarifocalLoss):
                cls_iou_targets = label_weights.new_zeros(cls_scores.shape)
                bg_class_ind = self.num_classes
                pos_inds = ((labels >= 0)
                            & (labels < bg_class_ind)).nonzero().squeeze(1)
                pos_labels = labels[pos_inds]
                if self.vfl_iou_type == 'mask':
                    pos_mask_preds = dn_mask_preds.detach()[mask_weights > 0]
                    pos_mask_preds = F.interpolate(
                        pos_mask_preds.unsqueeze(1),
                        scale_factor=2,
                        mode='bilinear',
                        align_corners=False).squeeze(1)
                    pos_mask_preds = (
                        pos_mask_preds > self.test_cfg.mask_thr_binary).float()

                    pos_mask_targets = mask_targets.float()
                    if pos_mask_preds.shape[-1] != pos_mask_targets.shape[-1] \
                            or pos_mask_preds.shape[-2] != pos_mask_targets.shape[-2]:
                        pos_mask_targets = F.interpolate(
                            pos_mask_targets.unsqueeze(1),
                            size=pos_mask_preds.shape[-2:],
                            mode='bilinear',
                            align_corners=False).squeeze(1)

                    cls_iou_targets[pos_inds, pos_labels] = mask_overlaps(
                        pos_mask_preds, pos_mask_targets)
                else:
                    pos_bbox_targets = bbox_targets[pos_inds]
                    pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
                    pos_bbox_pred = dn_bbox_preds.detach().reshape(-1, 4)[pos_inds]
                    pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
                    cls_iou_targets[pos_inds, pos_labels] = bbox_overlaps(
                        pos_decode_bbox_pred,
                        pos_decode_bbox_targets,
                        is_aligned=True)

                loss_cls = self.loss_cls(
                    cls_scores, cls_iou_targets, avg_factor=cls_avg_factor)
            else:
                loss_cls = self.loss_cls(
                    cls_scores,
                    labels,
                    label_weights,
                    avg_factor=cls_avg_factor)
        else:
            loss_cls = torch.zeros(
                1, dtype=cls_scores.dtype, device=cls_scores.device)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        if self.bg_cls_weight == 0:
            num_total_pos = cls_avg_factor
        else:
            num_total_pos = dn_bbox_preds.new_tensor([num_total_pos])
            num_total_pos = torch.clamp(
                reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = dn_mask_preds[mask_weights > 0]

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_pos)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_pos * self.num_points)

        return loss_cls, loss_bbox, loss_iou, loss_mask, loss_dice

    @torch.no_grad()
    def get_dn_targets(self, batch_gt_instances: InstanceList,
                       batch_img_metas: dict, dn_meta: Dict[str,
                                                            int]) -> Tuple:
        """Get targets in denoising part for a batch of images.

        Args:
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            tuple: a tuple containing the following targets.

            - labels_list (list[Tensor]): Labels for all images.
            - label_weights_list (list[Tensor]): Label weights for all images.
            - bbox_targets_list (list[Tensor]): BBox targets for all images.
            - bbox_weights_list (list[Tensor]): BBox weights for all images.
            - num_total_pos (int): Number of positive samples in all images.
            - num_total_neg (int): Number of negative samples in all images.
        """
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mask_targets_list, mask_weights_list, pos_inds_list,
         neg_inds_list) = multi_apply(
             self._get_dn_targets_single,
             batch_gt_instances,
             batch_img_metas,
             dn_meta=dn_meta)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, mask_targets_list, mask_weights_list,
                num_total_pos, num_total_neg)

    def _get_dn_targets_single(self, gt_instances: InstanceData,
                               img_meta: dict, dn_meta: Dict[str,
                                                             int]) -> Tuple:
        """Get targets in denoising part for one image.

        Args:
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_masks = gt_instances.masks
        num_groups = dn_meta['num_denoising_groups']
        num_denoising_queries = dn_meta['num_denoising_queries']
        num_queries_each_group = int(num_denoising_queries / num_groups)
        device = gt_bboxes.device

        if len(gt_labels) > 0:
            t = torch.arange(len(gt_labels), dtype=torch.long, device=device)
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = torch.arange(
                num_groups, dtype=torch.long, device=device)
            pos_inds = pos_inds.unsqueeze(1) * num_queries_each_group + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = \
                gt_bboxes.new_tensor([], dtype=torch.long)

        neg_inds = pos_inds + num_queries_each_group // 2

        # label targets
        labels = gt_bboxes.new_full((num_denoising_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_denoising_queries)

        # bbox targets
        bbox_targets = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        gt_bboxes_normalized = gt_bboxes / factor
        gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)
        bbox_targets[pos_inds] = gt_bboxes_targets.repeat([num_groups, 1])

        # mask targets
        mask_targets = gt_masks.repeat([num_groups, 1, 1])
        mask_weights = torch.zeros(num_denoising_queries, device=device)
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, bbox_targets, bbox_weights,
                mask_targets, mask_weights, pos_inds, neg_inds)


def mask_overlaps(masks1: Tensor, masks2: Tensor, eps: float = 1e-6):
    overlap = (masks1 * masks2).sum((-1, -2))
    area1 = masks1.sum((-1, -2))
    area2 = masks2.sum((-1, -2))
    union = area1 + area2 - overlap
    iou = overlap / (union + eps)
    return iou


@MODELS.register_module()
class MaskRTDETRHead_ppdet(RTDETRInsHead):

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             enc_outputs_mask: Tensor, init_outputs_class: Tensor,
             init_outputs_coord: Tensor, init_outputs_mask: Tensor,
             mask_features: Tensor, batch_data_samples: SampleList,
             dn_meta: Dict[str, int]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries_total,
                dim), where `num_queries_total` is the sum of
                `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries_total, 4) and each `inter_reference` has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            enc_outputs_class (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_outputs_coord (Tensor): The proposal generate from the
                encode feature map, has shape (bs, num_feat_points, 4) with the
                last dimension arranged as (cx, cy, w, h).
            enc_outputs_mask (Tensor): The mask generate from the
                encode feature map, has shape (bs, num_feat_points,
                mask_out_channels).
            mask_features (Tensor): instance mask features that
                has shape (bs, dim, h, w).
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            gt_instances = data_sample.gt_instances
            if isinstance(gt_instances.masks, BaseInstanceMasks):
                gt_instances.masks = gt_instances.masks.to_tensor(
                    dtype=torch.bool, device=mask_features.device)
            assert isinstance(gt_instances.masks, Tensor)
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references, mask_features)
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              enc_outputs_mask, init_outputs_class,
                              init_outputs_coord, init_outputs_mask,
                              batch_gt_instances, batch_img_metas, dn_meta)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        all_layers_mask_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        enc_mask_preds: Tensor,
        init_cls_scores: Tensor,
        init_bbox_preds: Tensor,
        init_mask_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
                group collation, including 'num_denoising_queries' and
                'num_denoising_groups'. It will be used for split outputs of
                denoising and matching parts and loss calculation.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        loss_dict = super().loss_by_feat(
            all_layers_cls_scores,
            all_layers_bbox_preds,
            all_layers_mask_preds,
            enc_cls_scores,
            enc_bbox_preds,
            enc_mask_preds,
            batch_gt_instances,
            batch_img_metas,
            dn_meta,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        if init_cls_scores is not None:
            # extract denoising and matching part of outputs
            (init_layers_matching_cls_scores, init_layers_matching_bbox_preds,
             init_layers_matching_mask_preds, init_layers_denoising_cls_scores,
             init_layers_denoising_bbox_preds,
             init_layers_denoising_mask_preds) = self.split_outputs(
                [init_cls_scores], [init_bbox_preds], [init_mask_preds],
                dn_meta)

            (init_losses_cls, init_losses_bbox, init_losses_iou,
             init_losses_mask, init_losses_dice) = multi_apply(
                self.loss_by_feat_single,
                init_layers_matching_cls_scores,
                init_layers_matching_bbox_preds,
                init_layers_matching_mask_preds,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas)
            loss_dict['init_loss_cls'] = init_losses_cls[-1]
            loss_dict['init_loss_bbox'] = init_losses_bbox[-1]
            loss_dict['init_loss_iou'] = init_losses_iou[-1]
            loss_dict['init_loss_mask'] = init_losses_mask[-1]
            loss_dict['init_loss_dice'] = init_losses_dice[-1]

            if init_layers_denoising_cls_scores is not None:
                # calculate denoising loss from all decoder layers
                (dn_init_losses_cls, dn_init_losses_bbox, dn_init_losses_iou,
                 dn_init_losses_mask, dn_init_losses_dice) = self.loss_dn(
                    init_layers_denoising_cls_scores,
                    init_layers_denoising_bbox_preds,
                    init_layers_denoising_mask_preds,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas,
                    dn_meta=dn_meta)
                # collate denoising loss
                loss_dict['dn_init_loss_cls'] = dn_init_losses_cls[-1]
                loss_dict['dn_init_loss_bbox'] = dn_init_losses_bbox[-1]
                loss_dict['dn_init_loss_iou'] = dn_init_losses_iou[-1]
                loss_dict['dn_init_loss_mask'] = dn_init_losses_mask[-1]
                loss_dict['dn_init_loss_dice'] = dn_init_losses_dice[-1]

        return loss_dict
