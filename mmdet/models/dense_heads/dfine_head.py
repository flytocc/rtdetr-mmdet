# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmengine.structures import InstanceData
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from mmdet.models.losses.utils import weighted_loss
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
from mmdet.structures.bbox.transforms import bbox_xyxy_to_cxcywh
from mmdet.utils import ConfigType, InstanceList, OptInstanceList, reduce_mean
from ..layers.transformer.dfine_layers import bbox2distance
from ..losses import VarifocalLoss
from ..utils import multi_apply
from .rtdetr_head import RTDETRHead


@MODELS.register_module()
class DFINEHead(RTDETRHead):
    r"""D-FINE: REDEFINE REGRESSION TASK IN DETRS
    AS FINE-GRAINED DISTRIBUTION REFINEMENT

    Code is modified from the `official github repo
    <https://github.com/Peterande/D-FINE>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2410.13842>`_ .
    """

    def __init__(self,
                 *args,
                 reg_max: int = 32,
                 reg_scale: float = 4,
                 layer_scale: float = 1.0,
                 eval_idx: int = -1,
                 share_pred_layer: bool = False,
                 num_pred_layer: int = 6,
                 fgl_loss_weight: float = 0.15,
                 loss_ld: ConfigType = dict(
                     type='KnowledgeDistillationKLDivLoss',
                     T=5,
                     reduction='none',
                     loss_weight=1.5),
                 **kwargs) -> None:
        assert not share_pred_layer
        self.reg_max = reg_max
        self.reg_scale = reg_scale
        self.layer_scale = layer_scale
        if eval_idx < 0:
            eval_idx = num_pred_layer - 1 + eval_idx
        self.eval_idx = eval_idx
        super().__init__(
            *args,
            share_pred_layer=False,
            num_pred_layer=num_pred_layer,
            **kwargs)
        self.fgl_loss_weight = fgl_loss_weight
        self.loss_ld = MODELS.build(loss_ld)

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

        cls_branches = [
            _gen_cls_branch(self.embed_dims, self.cls_out_channels)
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

        reg_branches = [
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

    @staticmethod
    def split_outputs(all_layers_cls_scores: List[Tensor],
                      all_layers_bbox_preds: List[Tensor],
                      all_layers_outputs_corners: List[Tensor],
                      dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
         all_layers_denoising_cls_scores,
         all_layers_denoising_bbox_preds) = RTDETRHead.split_outputs(
             all_layers_cls_scores, all_layers_bbox_preds, dn_meta)

        if dn_meta is not None:
            num_denoising_queries = dn_meta['num_denoising_queries']
            all_layers_denoising_bbox_corners = [
                o[:, :num_denoising_queries]
                for o in all_layers_outputs_corners
            ]
            all_layers_matching_bbox_corners = [
                o[:, num_denoising_queries:]
                for o in all_layers_outputs_corners
            ]
        else:
            all_layers_denoising_bbox_corners = None
            all_layers_matching_bbox_corners = all_layers_outputs_corners
        return (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
                all_layers_matching_bbox_corners,
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
                all_layers_denoising_bbox_corners)

    @torch.no_grad()
    def _get_match_indices(
            self, all_layers_matching_cls_scores: List[Tensor],
            all_layers_matching_bbox_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict]) -> List[List[Tuple[Tensor, Tensor]]]:
        """Get matching indices for all decoder layers."""
        num_imgs, num_queries, _ = all_layers_matching_cls_scores[0].shape
        gt_instances = InstanceData.cat(batch_gt_instances)
        num_target_list = list(map(len, batch_gt_instances))

        img_shapes = {
            tuple(img_meta['img_shape'])
            for img_meta in batch_img_metas
        }
        assert len(img_shapes) == 1, \
            f'All images must have the same shape, but got {img_shapes}.'

        img_meta = batch_img_metas[0]

        img_h, img_w = img_meta['img_shape']
        factor = all_layers_matching_bbox_preds[0].new_tensor(
            [img_w, img_h, img_w, img_h]).unsqueeze(0)

        all_layers_match_indices = []
        for cls_score, bbox_pred in zip(all_layers_matching_cls_scores,
                                        all_layers_matching_bbox_preds):
            if cls_score is None or bbox_pred is None:
                all_layers_match_indices.append(None)
                continue

            # batched calculate is more efficient
            cls_score = cls_score.flatten(0, 1)
            bbox_pred = bbox_pred.flatten(0, 1)

            # convert bbox_pred from xywh, normalized to xyxy, unnormalized
            bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
            bbox_pred = bbox_pred * factor

            pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred)

            total_cost = None
            for match_cost in self.assigner.match_costs:
                cost = match_cost(
                    pred_instances=pred_instances,
                    gt_instances=gt_instances,
                    img_meta=img_meta)
                total_cost = cost if total_cost is None else cost + total_cost
            total_cost = total_cost.view(num_imgs, num_queries, -1)

            batch_match_indices = []
            for bid, cost in enumerate(total_cost.split(num_target_list, -1)):
                row, col = linear_sum_assignment(cost[bid].cpu())
                batch_match_indices.append(
                    (torch.from_numpy(row).to(torch.long),
                     torch.from_numpy(col).to(torch.long)))

            all_layers_match_indices.append(batch_match_indices)

        return all_layers_match_indices

    @torch.no_grad()
    def _get_merged_match_indices(
        self, *all_match_indices: List[List[Tuple[Tensor, Tensor]]]
    ) -> List[Tuple[Tensor, Tensor]]:
        """Get a matching union set across all decoder layers."""
        results = []
        for all_indices in zip(*all_match_indices):
            all_indices = torch.cat(
                [torch.stack(inds, dim=-1) for inds in all_indices], dim=0)
            unique_pairs, counts = all_indices.unique(
                return_counts=True, dim=0)
            sorted_indices = counts.argsort(descending=True)
            unique_pairs = unique_pairs[sorted_indices]

            # Keep only the first (most frequent) gt_idx per anchor (row index)
            seen_rows = set()
            final_pairs = []
            for row_idx, col_idx in unique_pairs.tolist():
                if row_idx not in seen_rows:
                    seen_rows.add(row_idx)
                    final_pairs.append((row_idx, col_idx))

            if final_pairs:
                final_rows, final_cols = zip(*final_pairs)
                results.append((all_indices.new_tensor(final_rows),
                                all_indices.new_tensor(final_cols)))
            else:
                results.append(
                    (all_indices.new_empty(0), all_indices.new_empty(0)))

        return results

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
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
        all_layers_bbox_preds, all_layers_bbox_corners = all_layers_bbox_preds

        # extract denoising and matching part of outputs
        (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
         all_layers_matching_bbox_corners, all_layers_denoising_cls_scores,
         all_layers_denoising_bbox_preds,
         all_layers_denoising_bbox_corners) = self.split_outputs(
             all_layers_cls_scores, all_layers_bbox_preds,
             all_layers_bbox_corners, dn_meta)

        (*all_layers_match_indices,
         enc_match_indices) = self._get_match_indices(
             (*all_layers_matching_cls_scores, enc_cls_scores),
             (*all_layers_matching_bbox_preds, enc_bbox_preds),
             batch_gt_instances=batch_gt_instances,
             batch_img_metas=batch_img_metas)

        (initial_cls_scores,
         *all_layers_matching_cls_scores) = all_layers_matching_cls_scores
        (initial_bbox_preds,
         *all_layers_matching_bbox_preds) = all_layers_matching_bbox_preds
        (initial_match_indices,
         *all_layers_match_indices) = all_layers_match_indices

        # `_get_merged_match_indices` performs sorting，
        # be aware that the input order may influence training behavior.
        all_match_indices = (all_layers_match_indices[-1],
                             *all_layers_match_indices[:-1],
                             initial_match_indices)
        if enc_match_indices is not None:
            all_match_indices = (*all_match_indices, enc_match_indices)
        merged_match_indices = self._get_merged_match_indices(
            *all_match_indices)

        device = all_layers_matching_cls_scores[-1].device
        merged_match_indices = [(pos.to(device), gt.to(device))
                                for pos, gt in merged_match_indices]

        teacher_scores = all_layers_matching_cls_scores[-1].detach()
        teacher_corners = all_layers_matching_bbox_corners[-1].detach()
        teacher = (teacher_scores, teacher_corners)
        all_layers_teachers = (teacher, ) * (
            len(all_layers_matching_bbox_corners) - 1) + (None, )

        # initialize cached targets
        self.cached_bbox_targets = {}
        self.cached_fgl_targets = None
        self.cached_dn_targets = None
        self.cached_dn_fgl_targets = None
        self.num_pos, self.num_neg = None, None

        (losses_cls, losses_bbox, losses_iou, losses_fgl,
         losses_ddf) = multi_apply(
             self.loss_by_feat_single,
             all_layers_matching_cls_scores,
             all_layers_matching_bbox_preds,
             all_layers_matching_bbox_corners,
             all_layers_teachers,
             all_layers_match_indices,
             initial_bbox_preds=initial_bbox_preds.detach(),
             merged_match_indices=merged_match_indices,
             batch_gt_instances=batch_gt_instances,
             batch_img_metas=batch_img_metas)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_fgl'] = losses_fgl[-1]
        loss_dict['loss_ddf'] = losses_ddf[-1]
        # loss from other decoder layers
        for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i, loss_fgl_i,
                            loss_ddf_i) in \
                enumerate(zip(losses_cls[:-1], losses_bbox[:-1],
                              losses_iou[:-1], losses_fgl[:-1],
                              losses_ddf[:-1])):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_fgl'] = loss_fgl_i
            loss_dict[f'd{num_dec_layer}.loss_ddf'] = loss_ddf_i

        # loss of initial preds in decoder
        initial_loss_cls, initial_losses_bbox, initial_losses_iou = \
            self.loss_by_feat_single(
                initial_cls_scores, initial_bbox_preds,
                bbox_corners=None,
                teacher=None,
                batch_match_indices=initial_match_indices,
                initial_bbox_preds=None,
                merged_match_indices=merged_match_indices,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas)
        loss_dict['init_loss_cls'] = initial_loss_cls
        loss_dict['init_loss_bbox'] = initial_losses_bbox
        loss_dict['init_loss_iou'] = initial_losses_iou

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            # NOTE The enc_loss calculation of the DINO is
            # different from that of Deformable DETR.
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_by_feat_single(
                    enc_cls_scores, enc_bbox_preds,
                    bbox_corners=None,
                    teacher=None,
                    batch_match_indices=enc_match_indices,
                    initial_bbox_preds=None,
                    merged_match_indices=merged_match_indices,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        if all_layers_denoising_cls_scores is not None:
            (initial_dn_cls_scores,
            *all_layers_denoising_cls_scores) = all_layers_denoising_cls_scores
            (initial_dn_bbox_preds,
            *all_layers_denoising_bbox_preds) = all_layers_denoising_bbox_preds

            dn_teacher_scores = all_layers_denoising_cls_scores[-1].detach()
            dn_teacher_corners = all_layers_denoising_bbox_corners[-1].detach()
            dn_teacher = (dn_teacher_scores, dn_teacher_corners)
            all_layers_denoising_teachers = (dn_teacher, ) * (
                len(all_layers_denoising_bbox_corners) - 1) + (None, )

            # calculate denoising loss from all decoder layers
            (dn_losses_cls, dn_losses_bbox, dn_losses_iou, dn_losses_fgl,
             dn_losses_ddf) = multi_apply(
                 self._loss_dn_single,
                 all_layers_denoising_cls_scores,
                 all_layers_denoising_bbox_preds,
                 all_layers_denoising_bbox_corners,
                 all_layers_denoising_teachers,
                 initial_dn_bbox_preds=initial_dn_bbox_preds.detach(),
                 batch_gt_instances=batch_gt_instances,
                 batch_img_metas=batch_img_metas,
                 dn_meta=dn_meta)

            # collate denoising loss
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            loss_dict['dn_loss_fgl'] = dn_losses_fgl[-1]
            loss_dict['dn_loss_ddf'] = dn_losses_ddf[-1]
            for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i,
                                loss_fgl_i, loss_ddf_i) in \
                    enumerate(zip(dn_losses_cls[:-1], dn_losses_bbox[:-1],
                                  dn_losses_iou[:-1], dn_losses_fgl[:-1],
                                  dn_losses_ddf[:-1])):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
                loss_dict[f'd{num_dec_layer}.dn_loss_fgl'] = loss_fgl_i
                loss_dict[f'd{num_dec_layer}.dn_loss_ddf'] = loss_ddf_i

            # loss of initial preds in decoder
            initial_loss_cls, initial_losses_bbox, initial_losses_iou, _, _ = \
                self._loss_dn_single(
                    initial_dn_cls_scores, initial_dn_bbox_preds,
                    dn_bbox_corners=None,
                    teacher=None,
                    initial_dn_bbox_preds=None,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas,
                    dn_meta=dn_meta)
            loss_dict['dn_init_loss_cls'] = initial_loss_cls
            loss_dict['dn_init_loss_bbox'] = initial_losses_bbox
            loss_dict['dn_init_loss_iou'] = initial_losses_iou

        return loss_dict

    def loss_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor,
                            bbox_corners: Optional[Tensor],
                            teacher: Optional[Tuple[Tensor, Tensor]],
                            batch_match_indices: Optional[List[Tuple[Tensor,
                                                                     Tensor]]],
                            initial_bbox_preds: Optional[Tensor],
                            merged_match_indices: List[Tuple[Tensor, Tensor]],
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
            bbox_corners (Tensor):
                # TODO
            teacher (tuple[Tensor, Tensor]):
                # TODO
            batch_match_indices (list[tuple[Tensor, Tensor]]):
                # TODO
            initial_bbox_preds (Tensor):
                # TODO
            merged_match_indices (list[tuple[Tensor, Tensor]]):
                # TODO
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        num_imgs, num_queries, _ = cls_scores.shape
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_num_pos_list) = multi_apply(
             self._get_cls_targets_single,
             batch_match_indices,
             batch_gt_instances,
             batch_img_metas,
             num_queries=num_queries,
             device=bbox_preds.device)
        num_total_pos = sum(bbox_num_pos_list)
        num_total_neg = num_imgs * num_queries - num_total_pos
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, VarifocalLoss):
            bg_class_ind = self.num_classes
            pos_inds = ((labels >= 0)
                        & (labels < bg_class_ind)).nonzero().squeeze(1)
            cls_iou_targets = cls_scores.new_zeros(cls_scores.shape)
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
            pos_bbox_pred = bbox_preds.reshape(-1, 4)[pos_inds]
            pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
            pos_labels = labels[pos_inds]
            cls_iou_targets[pos_inds, pos_labels] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True).type_as(cls_iou_targets)
            loss_cls = self.loss_cls(
                cls_scores, cls_iou_targets, avg_factor=cls_avg_factor)
        else:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        if num_queries not in self.cached_bbox_targets:
            (bbox_targets_list, bbox_weights_list,
             bbox_num_pos_list) = multi_apply(
                 self._get_bbox_targets_single,
                 merged_match_indices,
                 batch_gt_instances,
                 batch_img_metas,
                 num_queries=num_queries,
                 device=bbox_preds.device)
            num_total_bbox_pos = sum(bbox_num_pos_list)
            bbox_targets = torch.cat(bbox_targets_list, 0)
            bbox_weights = torch.cat(bbox_weights_list, 0)

            # Compute the average number of gt boxes across all gpus, for
            # normalization purposes
            bbox_avg_factor = bbox_preds.new_tensor([num_total_bbox_pos])
            bbox_avg_factor = torch.clamp(
                reduce_mean(bbox_avg_factor), min=1).item()

            self.cached_bbox_targets[num_queries] = (bbox_targets,
                                                     bbox_weights,
                                                     num_total_bbox_pos,
                                                     bbox_avg_factor)
        else:
            # use cached bbox targets
            (bbox_targets, bbox_weights, num_total_bbox_pos,
             bbox_avg_factor) = self.cached_bbox_targets[num_queries]

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
            bboxes, bboxes_gt, bbox_weights, avg_factor=bbox_avg_factor)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=bbox_avg_factor)

        if bbox_corners is None:
            return loss_cls, loss_bbox, loss_iou

        bbox_pos_inds = torch.nonzero(
            bbox_weights.sum(-1) > 0, as_tuple=False).squeeze(-1).unique()

        # distribution focal loss
        initial_bbox_preds = initial_bbox_preds.reshape(-1, 4)
        bbox_corners = bbox_corners.reshape(-1, 4, self.reg_max + 1)

        if self.cached_fgl_targets is None:
            self.cached_fgl_targets = bbox2distance(
                initial_bbox_preds[bbox_pos_inds],
                bbox_cxcywh_to_xyxy(bbox_targets[bbox_pos_inds]), self.reg_max,
                self.reg_scale, 0.5)
        target_corners, weight_right, weight_left = self.cached_fgl_targets

        pos_ious = bbox_overlaps(
            bboxes[bbox_pos_inds], bboxes_gt[bbox_pos_inds],
            is_aligned=True).detach()
        weight_targets = pos_ious.unsqueeze(-1).repeat(1, 4).reshape(-1)

        loss_fgl = self.fgl_loss_weight * unimodal_distribution_focal_loss(
            bbox_corners[bbox_pos_inds].reshape(-1, self.reg_max + 1),
            target_corners,
            weight_right=weight_right,
            weight_left=weight_left,
            weight=weight_targets,
            avg_factor=bbox_avg_factor)

        # vari KnowledgeDistillationKLDivLoss
        if teacher is not None:
            teacher_scores, teacher_corners = teacher
            teacher_scores = teacher_scores.reshape(-1, self.cls_out_channels)
            teacher_corners = teacher_corners.reshape(-1, self.reg_max + 1)
            bbox_corners = bbox_corners.reshape(-1, self.reg_max + 1)

            weight_targets_local = teacher_scores.sigmoid().max(dim=-1)[0]
            weight_targets_local[bbox_pos_inds] = \
                pos_ious.type_as(weight_targets_local)
            weight_targets_local = \
                weight_targets_local.unsqueeze(-1).repeat(1, 4).reshape(-1)

            loss_match_local = self.loss_ld(bbox_corners, teacher_corners,
                                            weight_targets_local) * (
                                                self.reg_max + 1)

            mask = bbox_weights.bool().reshape(-1)
            num_total_bbox_neg = bbox_weights.size(0) - num_total_bbox_pos
            if self.num_pos is None:
                self.num_pos = (num_total_bbox_pos * 4 * 8 / num_imgs)**0.5
                self.num_neg = (num_total_bbox_neg * 4 * 8 / num_imgs)**0.5
            loss_match_local1 = loss_match_local[mask].mean() \
                if num_total_bbox_pos > 0 else 0
            loss_match_local2 = loss_match_local[~mask].mean() \
                if num_total_bbox_neg > 0 else 0
            loss_ddf = (loss_match_local1 * self.num_pos +
                        loss_match_local2 * self.num_neg) / (
                            self.num_pos + self.num_neg)
        else:
            loss_ddf = bbox_corners.new_tensor(0)

        return loss_cls, loss_bbox, loss_iou, loss_fgl, loss_ddf

    @torch.no_grad()
    def _get_cls_targets_single(self, match_indices: Tuple[Tensor, Tensor],
                                gt_instances: InstanceData, img_meta: dict,
                                num_queries: int,
                                device: Union[str, torch.device]) -> tuple:
        """Compute classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            match_indices (tuple[Tensor, Tensor]):
                A tuple containing two tensors, the first is the sampled
                positive indices for each image, and the second is the
                assigned ground truth indices for each positive sample.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.
            num_queries (int): The number of queries for the current image.
            device (str or torch.device): The device of the output tensors.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - num_pos (int): The number of positive samples for the image.
        """
        pos_inds, pos_assigned_gt_inds = match_indices
        gt_bboxes = gt_instances.bboxes
        dtype = gt_bboxes.dtype

        img_h, img_w = img_meta['img_shape']
        factor = torch.tensor([img_w, img_h, img_w, img_h],
                              dtype=dtype,
                              device=device).unsqueeze(0)

        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds]
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)

        bbox_targets = torch.zeros((num_queries, 4),
                                   dtype=dtype,
                                   device=device)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        # label targets
        gt_labels = gt_instances.labels
        labels = torch.full((num_queries, ),
                            self.num_classes,
                            dtype=torch.long,
                            device=device)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones(num_queries)

        return labels, label_weights, bbox_targets, pos_inds.numel()

    @torch.no_grad()
    def _get_bbox_targets_single(self, match_indices: Tuple[Tensor, Tensor],
                                 gt_instances: InstanceData, img_meta: dict,
                                 num_queries: int,
                                 device: Union[str, torch.device]) -> tuple:
        """Compute regression targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            match_indices (tuple[Tensor, Tensor]):
                A tuple containing two tensors, the first is the sampled
                positive indices for each image, and the second is the
                assigned ground truth indices for each positive sample.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.
            num_queries (int): The number of queries for the current image.
            device (str or torch.device): The device of the output tensors.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - num_pos (int): The number of positive samples for the image.
        """
        pos_inds, pos_assigned_gt_inds = match_indices
        gt_bboxes = gt_instances.bboxes
        dtype = gt_bboxes.dtype

        img_h, img_w = img_meta['img_shape']
        factor = torch.tensor([img_w, img_h, img_w, img_h],
                              dtype=dtype,
                              device=device).unsqueeze(0)

        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds]
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)

        bbox_targets = torch.zeros((num_queries, 4),
                                   dtype=dtype,
                                   device=device)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        # bbox weights
        bbox_weights = torch.zeros((num_queries, 4),
                                   dtype=dtype,
                                   device=device)
        bbox_weights[pos_inds] = 1.0

        return bbox_targets, bbox_weights, pos_inds.numel()

    def _loss_dn_single(self, dn_cls_scores: Tensor, dn_bbox_preds: Tensor,
                        dn_bbox_corners: Optional[Tensor],
                        teacher: Optional[Tuple[Tensor, Tensor]],
                        initial_dn_bbox_preds: Optional[Tensor],
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
            dn_bbox_corners (Tensor):
                # TODO
            teacher (tuple[Tensor, Tensor]):
                # TODO
            initial_dn_bbox_preds (Tensor):
                # TODO
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
        if dn_cls_scores.size(1) == 0:
            loss_cls = dn_cls_scores.new_tensor(0)
            loss_bbox = loss_iou = dn_bbox_preds.new_tensor(0)
            loss_fgl = loss_ddf = dn_bbox_corners.new_tensor(0) \
                if dn_bbox_corners is not None else None
            return loss_cls, loss_bbox, loss_iou, loss_fgl, loss_ddf

        if self.cached_dn_targets is None:
            cls_reg_targets = self.get_dn_targets(batch_gt_instances,
                                                  batch_img_metas, dn_meta)
            (labels_list, label_weights_list, bbox_targets_list,
             bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets
            labels = torch.cat(labels_list, 0)
            label_weights = torch.cat(label_weights_list, 0)
            bbox_targets = torch.cat(bbox_targets_list, 0)
            bbox_weights = torch.cat(bbox_weights_list, 0)

            # construct weighted avg_factor to match with the official DETR repo
            cls_avg_factor = \
                num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
            if self.sync_cls_avg_factor:
                cls_avg_factor = reduce_mean(
                    dn_bbox_preds.new_tensor([cls_avg_factor]))
            cls_avg_factor = max(cls_avg_factor, 1)

            # Compute the average number of gt boxes across all gpus, for
            # normalization purposes
            bbox_avg_factor = dn_bbox_preds.new_tensor([num_total_pos])
            bbox_avg_factor = torch.clamp(
                reduce_mean(bbox_avg_factor), min=1).item()

            self.cached_dn_targets = (labels, label_weights, bbox_targets,
                                      bbox_weights, num_total_pos,
                                      cls_avg_factor, bbox_avg_factor)
        else:
            # use cached dn targets
            (labels, label_weights, bbox_targets, bbox_weights, num_total_pos,
             cls_avg_factor, bbox_avg_factor) = self.cached_dn_targets

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)

        if isinstance(self.loss_cls, VarifocalLoss):
            bg_class_ind = self.num_classes
            pos_inds = ((labels >= 0)
                        & (labels < bg_class_ind)).nonzero().squeeze(1)
            cls_iou_targets = cls_scores.new_zeros(cls_scores.shape)
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
            pos_bbox_pred = dn_bbox_preds.reshape(-1, 4)[pos_inds]
            pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
            pos_labels = labels[pos_inds]
            cls_iou_targets[pos_inds, pos_labels] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True).type_as(cls_iou_targets)
            loss_cls = self.loss_cls(
                cls_scores, cls_iou_targets, avg_factor=cls_avg_factor)
        else:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

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
            bboxes, bboxes_gt, bbox_weights, avg_factor=bbox_avg_factor)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=bbox_avg_factor)

        if dn_bbox_corners is None:
            return loss_cls, loss_bbox, loss_iou, None, None

        bbox_pos_inds = torch.nonzero(
            bbox_weights.sum(-1) > 0, as_tuple=False).squeeze(-1).unique()

        # distribution focal loss
        initial_dn_bbox_preds = initial_dn_bbox_preds.reshape(-1, 4)
        dn_bbox_corners = dn_bbox_corners.reshape(-1, 4, self.reg_max + 1)

        if self.cached_dn_fgl_targets is None:
            self.cached_dn_fgl_targets = bbox2distance(
                initial_dn_bbox_preds[bbox_pos_inds],
                bbox_cxcywh_to_xyxy(bbox_targets[bbox_pos_inds]), self.reg_max,
                self.reg_scale, 0.5)
        target_corners, weight_right, weight_left = self.cached_dn_fgl_targets

        pos_ious = bbox_overlaps(
            bboxes[bbox_pos_inds], bboxes_gt[bbox_pos_inds],
            is_aligned=True).detach()
        weight_targets = pos_ious.unsqueeze(-1).repeat(1, 4).reshape(-1)

        loss_fgl = self.fgl_loss_weight * unimodal_distribution_focal_loss(
            dn_bbox_corners[bbox_pos_inds].reshape(-1, self.reg_max + 1),
            target_corners,
            weight_right=weight_right,
            weight_left=weight_left,
            weight=weight_targets,
            avg_factor=bbox_avg_factor)

        # vari KnowledgeDistillationKLDivLoss
        if teacher is not None:
            teacher_scores, teacher_corners = teacher
            teacher_scores = teacher_scores.reshape(-1, self.cls_out_channels)
            teacher_corners = teacher_corners.reshape(-1, self.reg_max + 1)
            dn_bbox_corners = dn_bbox_corners.reshape(-1, self.reg_max + 1)

            weight_targets_local = teacher_scores.sigmoid().max(dim=-1)[0]
            weight_targets_local[bbox_pos_inds] = \
                pos_ious.type_as(weight_targets_local)
            weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(
                1, 4).reshape(-1)

            loss_match_local = self.loss_ld(dn_bbox_corners, teacher_corners,
                                            weight_targets_local) * (
                                                self.reg_max + 1)

            mask = bbox_weights.bool().reshape(-1)
            num_total_bbox_pos = num_total_pos
            num_total_bbox_neg = bbox_weights.size(0) - num_total_bbox_pos
            loss_match_local1 = loss_match_local[mask].mean() \
                if num_total_bbox_pos > 0 else 0
            loss_match_local2 = loss_match_local[~mask].mean() \
                if num_total_bbox_neg > 0 else 0
            loss_ddf = (loss_match_local1 * self.num_pos +
                        loss_match_local2 * self.num_neg) / (
                            self.num_pos + self.num_neg)
        else:
            loss_ddf = dn_bbox_corners.new_tensor(0)

        return loss_cls, loss_bbox, loss_iou, loss_fgl, loss_ddf


@weighted_loss
def unimodal_distribution_focal_loss(pred, label, weight_right, weight_left):
    dis_left = label.long()
    dis_right = dis_left + 1
    loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left \
        + F.cross_entropy(pred, dis_right, reduction='none') * weight_right
    return loss
