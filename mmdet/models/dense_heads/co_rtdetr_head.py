# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.utils import multi_apply
from mmdet.registry import MODELS
from mmdet.structures.bbox import (bbox_cxcywh_to_xyxy, bbox_overlaps,
                                   bbox_xyxy_to_cxcywh)
from mmdet.utils import OptInstanceList, reduce_mean
from ..losses import RTDETRVarifocalLoss
from .dfine_head import DFINEHead
from .rtdetr_head import RTDETRHead


class CoRTDETRHeadMixin:

    def get_aux_targets(self, pos_coords, max_pos_coords, batch_img_metas):
        coords, labels, targets, all_feats = pos_coords

        all_pos_inds = []
        for label in labels:
            bg_class_ind = self.num_classes
            pos_inds = ((label >= 0)
                        & (label < bg_class_ind)).nonzero().squeeze(1)
            all_pos_inds.append(pos_inds)
        max_num_coords = max(map(len, all_pos_inds))
        max_num_coords = max(min(max_num_coords, max_pos_coords), 9)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         coords_list, feats_list) = multi_apply(
             self._get_aux_targets_single,
             coords,
             labels,
             targets,
             all_feats,
             all_pos_inds,
             batch_img_metas,
             max_num_coords=max_num_coords)
        gt_instances = InstanceData()
        gt_instances.labels = torch.cat(labels_list, 0)
        gt_instances.label_weights = torch.cat(label_weights_list, 0)
        gt_instances.bbox_targets = torch.cat(bbox_targets_list, 0)
        gt_instances.bbox_weights = torch.cat(bbox_weights_list, 0)

        aux_input_dict = dict(
            aux_coords=torch.stack(coords_list, 0),
            aux_feats=torch.stack(feats_list, 0))
        return gt_instances, aux_input_dict

    def _get_aux_targets_single(self, coord: Tensor, label: Tensor,
                                target: Tensor, feats: Tensor,
                                pos_inds: Tensor, img_meta: dict,
                                max_num_coords: int) -> tuple:
        if pos_inds.shape[0] > max_num_coords:
            indices = torch.randperm(
                pos_inds.shape[0], dtype=pos_inds.dtype)[:max_num_coords]
            pos_inds = pos_inds[indices]
            inds = pos_inds
        elif pos_inds.shape[0] < max_num_coords:
            bg_class_ind = self.num_classes
            neg_inds = (label == bg_class_ind).nonzero().squeeze(1)
            padding_shape = max_num_coords - pos_inds.shape[0]
            indices = torch.randperm(
                neg_inds.shape[0], dtype=neg_inds.dtype)[:padding_shape]
            neg_inds = neg_inds[indices]
            inds = torch.cat([pos_inds, neg_inds], dim=0)
        else:
            inds = pos_inds

        num_coords_per_point = coord.shape[0] // feats.shape[0]
        feats = feats.unsqueeze(1).repeat(1, num_coords_per_point, 1)
        feats = feats.reshape(feats.shape[0] * num_coords_per_point, -1)

        coord = coord[inds]
        label = label[inds]
        target = target[inds]
        feats = feats[inds]

        img_h, img_w = img_meta['img_shape']
        factor = coord.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        coord = bbox_xyxy_to_cxcywh(coord / factor)
        target = bbox_xyxy_to_cxcywh(target / factor)

        label_weights = coord.new_ones([max_num_coords])
        bbox_weights = coord.new_zeros([max_num_coords, 4])
        bbox_weights[:pos_inds.shape[0]] = 1
        return (label, label_weights, target, bbox_weights, coord, feats)

    def loss_aux(self, hidden_states: Tensor, references: List[Tensor],
                 gt_instances: InstanceData,
                 batch_img_metas: List[dict]) -> dict:
        outs = self(hidden_states, references)
        loss_inputs = outs + (gt_instances, batch_img_metas)
        losses = self.loss_aux_by_feat(*loss_inputs)
        return losses

    def loss_aux_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        gt_instances: InstanceData,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        assert batch_gt_instances_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            'for batch_gt_instances_ignore setting to None.'

        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_aux_by_feat_single,
            all_layers_cls_scores,
            all_layers_bbox_preds,
            gt_instances=gt_instances,
            batch_img_metas=batch_img_metas)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['aux_loss_cls'] = losses_cls[-1]
        loss_dict['aux_loss_bbox'] = losses_bbox[-1]
        loss_dict['aux_loss_iou'] = losses_iou[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.aux_loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.aux_loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.aux_loss_iou'] = loss_iou_i
            num_dec_layer += 1
        return loss_dict

    def loss_aux_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor,
                                gt_instances: InstanceData,
                                batch_img_metas: List[dict]) -> Tuple[Tensor]:
        num_queries = cls_scores.size(1)

        if num_queries == 0:
            loss_cls = cls_scores.mean() * 0
            loss_bbox = loss_iou = bbox_preds.mean() * 0
            return (loss_cls, loss_bbox, loss_iou)

        labels = gt_instances.labels
        label_weights = gt_instances.label_weights
        bbox_targets = gt_instances.bbox_targets
        bbox_weights = gt_instances.bbox_weights

        bg_class_ind = self.num_classes
        num_total_pos = len(
            ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1))
        num_imgs = cls_scores.size(0)
        num_total_neg = num_imgs * num_queries - num_total_pos

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, RTDETRVarifocalLoss):
            bg_class_ind = self.num_classes
            pos_inds = ((labels >= 0)
                        & (labels < bg_class_ind)).nonzero().squeeze(1)
            cls_iou_targets = label_weights.new_zeros(cls_scores.shape)
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
            pos_bbox_pred = bbox_preds.reshape(-1, 4)[pos_inds]
            pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
            pos_labels = labels[pos_inds]
            cls_iou_targets[pos_inds, pos_labels] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            loss_cls = self.loss_cls(
                cls_scores, cls_iou_targets, avg_factor=cls_avg_factor)
        else:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w = img_meta['img_shape']
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
        return loss_cls, loss_bbox, loss_iou


@MODELS.register_module()
class CoRTDETRHead(CoRTDETRHeadMixin, RTDETRHead):
    """A CoRTDETRHeadMixin vision of RTDETRHead"""


@MODELS.register_module()
class CoDFINEHead(CoRTDETRHeadMixin, DFINEHead):
    """A CoRTDETRHeadMixin vision of DFINEHead"""
