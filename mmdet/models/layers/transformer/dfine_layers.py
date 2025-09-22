# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from copy import deepcopy
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.config import ConfigDict
from mmengine.model import (BaseModule, ModuleList, bias_init_with_prob,
                            constant_init, xavier_init)
from torch import Tensor, nn

from mmdet.models.layers.transformer import \
    DeformableDetrTransformerDecoderLayer
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .dino_layers import CdnQueryGenerator
from .rtdetr_layers import RTDETRFPN, CSPLayer, RTDETRTransformerDecoder
from .utils import MLP, inverse_sigmoid


class RepNCSPELAN4(BaseModule):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: float = 1.0,
                 num_blocks: int = 3,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        mid_channels = int(out_channels * expand_ratio // 2)
        self.cv1 = ConvModule(
            in_channels,
            in_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.cv2 = nn.Sequential(
            CSPLayer(
                in_channels // 2,
                mid_channels,
                expand_ratio=1.0,
                num_blocks=num_blocks,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                mid_channels,
                mid_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        self.cv3 = nn.Sequential(
            CSPLayer(
                mid_channels,
                mid_channels,
                expand_ratio=1.0,
                num_blocks=num_blocks,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                mid_channels,
                mid_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        self.cv4 = ConvModule(
            in_channels + mid_channels * 2,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.cv1(x).chunk(2, dim=1)
        x3 = self.cv2(x2)
        x4 = self.cv3(x3)
        return self.cv4(torch.cat([x1, x2, x3, x4], dim=1))


@MODELS.register_module()
class DFINEFPN(RTDETRFPN):
    """FPN of D-FINE.

    Args:
        in_channels (List[int], optional): The input channels of the
            feature maps. Defaults to [256, 256, 256].
        out_channels (int, optional): The output dimension of the MLP.
            Defaults to 256.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer.
            Defaults to 3.
        expansion (float, optional): The expansion of the CSPLayer.
            Defaults to 1.0.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (:obj:`ConfigDict` or dict, optional): The config dict for
            normalization layers. Defaults to dict(type='BN').
        act_cfg (:obj:`ConfigDict` or dict, optional): The config dict for
            activation layers. Defaults to dict(type='SiLU', inplace=True).
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`], optional): Initialization config dict.
    """

    def __init__(
        self,
        in_channels: List[int] = [256, 256, 256],
        out_channels: int = 256,
        num_csp_blocks: int = 3,
        expansion: float = 1.0,
        upsample_cfg: ConfigType = dict(scale_factor=2, mode='nearest'),
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='SiLU', inplace=True),
        init_cfg: OptMultiConfig = dict(
            type='Kaiming',
            layer='Conv2d',
            a=math.sqrt(5),
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # top-down fpn
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None))
            self.top_down_blocks.append(
                RepNCSPELAN4(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    expand_ratio=expansion,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                nn.Sequential(
                    ConvModule(
                        in_channels[idx],
                        in_channels[idx],
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=None),
                    ConvModule(
                        in_channels[idx],
                        in_channels[idx],
                        3,
                        stride=2,
                        padding=1,
                        groups=in_channels[idx],
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=None)))
            self.bottom_up_blocks.append(
                RepNCSPELAN4(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    expand_ratio=expansion,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None) if in_channels[i] != out_channels else nn.
                Identity())


@lru_cache
def weighting_function(reg_max: int,
                       up: float,
                       reg_scale: float,
                       device: str = 'cpu',
                       dtype: torch.dtype = torch.float) -> Tensor:
    """Generates the non-uniform Weighting Function W(n) for bounding box
    regression.

    Args:
        reg_max (int): Max number of the discrete bins.
        up (float): Controls upper bounds of the sequence,
            where maximum offset is ±up * H / W.
        reg_scale (float): Controls the curvature of the Weighting Function.
            Larger values result in flatter weights near the central axis
            W(reg_max/2)=0 and steeper weights at both ends.

    Returns:
        Tensor: Sequence of Weighting Function.
    """
    upper_bound1 = abs(up) * abs(reg_scale)
    upper_bound2 = abs(up) * abs(reg_scale) * 2
    step = (upper_bound1 + 1)**(2 / (reg_max - 2))
    left_values = [-(step)**i + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right_values = [(step)**i - 1 for i in range(1, reg_max // 2)]
    project = [-upper_bound2, *left_values, 0, *right_values, upper_bound2]
    return torch.Tensor(project).to(device, dtype)


def translate_gt(gt, reg_max, reg_scale, up):
    """Decodes bounding box ground truth (GT) values into distribution-based GT
    representations.

    This function maps continuous GT values into discrete distribution bins,
    which can be used for regression tasks in object detection models. It
    calculates the indices of the closest bins to each GT value and assigns
    interpolation weights to these bins based on their proximity to the GT
    value.

    Args:
        gt (Tensor): Ground truth bounding box values, shape (N, ).
        reg_max (int): Maximum number of discrete bins for the distribution.
        reg_scale (float): Controls the curvature of the Weighting Function.
        up (Tensor): Controls the upper bounds of the Weighting Function.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            - indices (Tensor): Index of the left bin closest to each GT value,
              shape (N, ).
            - weight_right (Tensor): Weight assigned to the right bin,
              shape (N, ).
            - weight_left (Tensor): Weight assigned to the left bin,
              shape (N, ).
    """
    gt = gt.reshape(-1)
    function_values = weighting_function(reg_max, up, reg_scale, gt.device,
                                         gt.dtype)

    # Find the closest left-side indices for each value
    diffs = function_values.unsqueeze(0) - gt.unsqueeze(1)
    mask = diffs <= 0
    closest_left_indices = torch.sum(mask, dim=1) - 1

    # Calculate the weights for the interpolation
    indices = closest_left_indices.float()

    weight_right = torch.zeros_like(indices)
    weight_left = torch.zeros_like(indices)

    valid_idx_mask = (indices >= 0) & (indices < reg_max)
    valid_indices = indices[valid_idx_mask].long()

    # Obtain distances
    left_values = function_values[valid_indices]
    right_values = function_values[valid_indices + 1]

    left_diffs = torch.abs(gt[valid_idx_mask] - left_values)
    right_diffs = torch.abs(right_values - gt[valid_idx_mask])

    # Valid weights
    weight_right[valid_idx_mask] = left_diffs / (left_diffs + right_diffs)
    weight_left[valid_idx_mask] = 1.0 - weight_right[valid_idx_mask]

    # Invalid weights (out of range)
    invalid_idx_mask_neg = (indices < 0)
    weight_right[invalid_idx_mask_neg] = 0.0
    weight_left[invalid_idx_mask_neg] = 1.0
    indices[invalid_idx_mask_neg] = 0.0

    invalid_idx_mask_pos = (indices >= reg_max)
    weight_right[invalid_idx_mask_pos] = 1.0
    weight_left[invalid_idx_mask_pos] = 0.0
    indices[invalid_idx_mask_pos] = reg_max - 0.1

    return indices, weight_right, weight_left


@torch.no_grad()
def bbox2distance(points, bbox, reg_max, reg_scale, up, eps=0.1):
    """Converts bounding box coordinates to distances from a reference point.

    Args:
        points (Tensor): (n, 4) [x, y, w, h], where (x, y) is the center.
        bbox (Tensor): (n, 4) bounding boxes in "xyxy" format.
        reg_max (float): Maximum bin value.
        reg_scale (float): Controlling curvarture of W(n).
        up (Tensor): Controlling upper bounds of W(n).
        eps (float): Small value to ensure target < reg_max.

    Returns:
        Tensor: Decoded distances.
    """
    s = points[..., 2:] + 1e-16
    lt = (points[..., :2] - bbox[..., :2]) / s
    rb = (bbox[..., 2:] - points[..., :2]) / s
    four_lens = torch.cat([lt, rb], dim=-1)
    four_lens = (four_lens - 0.5) * abs(reg_scale)

    four_lens, weight_right, weight_left = translate_gt(
        four_lens, reg_max, reg_scale, up)
    if reg_max is not None:
        four_lens = four_lens.clamp(min=0, max=reg_max - eps)
    return four_lens.reshape(
        -1).detach(), weight_right.detach(), weight_left.detach()


def distance2bbox(points, distance, reg_scale, clamp_wh=True):
    """Decodes edge-distances into bounding box coordinates.

    Args:
        points (Tensor): (B, N, 4) or (N, 4) format, representing [x, y, w, h],
            where (x, y) is the center and (w, h) are width and height.
        distance (Tensor): (B, N, 4) or (N, 4), representing distances from the
            point to the left, top, right, and bottom boundaries.
        reg_scale (float): Controls the curvature of the Weighting Function.

    Returns:
        Tensor: Bounding boxes in (N, 4) or (B, N, 4) format [cx, cy, w, h].
    """
    distance = distance / abs(reg_scale) + 0.5
    cxcy, wh = points[..., :2], points[..., 2:]
    x1y1 = cxcy - distance[..., :2] * wh
    x2y2 = cxcy + distance[..., 2:] * wh
    decoded_cxcy = (x1y1 + x2y2) / 2
    decoded_wh = x2y2 - x1y1
    if clamp_wh:
        decoded_wh = decoded_wh.clamp(min=0)
    return torch.cat([decoded_cxcy, decoded_wh], dim=-1)


class Gate(BaseModule):

    def __init__(self, embed_dims: int) -> None:
        super().__init__()
        self.gate = nn.Linear(2 * embed_dims, 2 * embed_dims)

    def init_weights(self) -> None:
        bias = bias_init_with_prob(0.5)
        nn.init.constant_(self.gate.bias, bias)
        nn.init.zeros_(self.gate.weight)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return gate1 * x1 + gate2 * x2


def multi_num_points_multi_scale_deformable_attn_pytorch(
        value: torch.Tensor, value_spatial_shapes: torch.Tensor,
        sampling_locations: torch.Tensor, attention_weights: torch.Tensor,
        num_points_list: List[int]) -> torch.Tensor:
    """CPU version of multi-num_points multi-scale deformable attention.

    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs, num_queries, num_heads, total_num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs, num_queries, num_heads, total_num_points),

    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, total_num_points, _ =\
        sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                             dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_grids_list = sampling_grids.split(num_points_list, dim=-2)
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids_list[level].transpose(1, 2).flatten(
            0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, total_num_points) ->
    # (bs, num_heads, num_queries, total_num_points) ->
    # (bs*num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, total_num_points)
    output = (torch.cat(sampling_value_list, dim=-1) *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims,
                                              num_queries)
    return output.transpose(1, 2).contiguous()


class MultiNumPointsMultiScaleDeformableAttention(BaseModule):
    """MultiScaleDeformableAttention of multi num points.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 8.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int|list[int]): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        value_proj_ratio (float): The expansion ratio of value_proj.
            Default: 1.0.
    """

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 num_levels: int = 4,
                 num_points: Union[int, Tuple[int]] = 4,
                 im2col_step: int = 64,
                 dropout: float = 0.1,
                 batch_first: bool = False,
                 norm_cfg: Optional[dict] = None,
                 init_cfg: Optional[ConfigDict] = None,
                 value_proj_ratio: float = 1.0):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads

        if isinstance(num_points, int):
            num_levels = [num_points] * num_levels
        assert isinstance(num_points, tuple) and len(num_points) == num_levels

        num_points_scale = [1 / n for n in num_points for _ in range(n)]
        self.register_buffer(
            'num_points_scale',
            torch.tensor(num_points_scale, dtype=torch.float32).unsqueeze(-1),
            persistent=False)

        self.total_num_points = sum(num_points)
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * self.total_num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * self.total_num_points)
        value_proj_size = int(embed_dims * value_proj_ratio)
        self.value_proj = nn.Linear(embed_dims, value_proj_size)
        self.output_proj = nn.Linear(value_proj_size, embed_dims)
        self.init_weights()

    def init_weights(self) -> None:
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        device = next(self.parameters()).device
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32,
            device=device) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 2).repeat(1, self.total_num_points,
                                                      1)
        grid_init *= torch.cat([
            torch.arange(1, n + 1, device=device) for n in self.num_points
        ]).view(1, -1, 1)

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                identity: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                reference_points: Optional[torch.Tensor] = None,
                spatial_shapes: Optional[torch.Tensor] = None,
                level_start_index: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (torch.Tensor): The positional encoding for `query`.
                Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.total_num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.total_num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.total_num_points)

        reference_points = self._expand_to_total_num_points(reference_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            offset_normalizer = self._expand_to_total_num_points(
                offset_normalizer)

            sampling_locations = reference_points[:, :, None, :, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, :2] \
                + sampling_offsets * self.num_points_scale \
                * reference_points[:, :, None, :, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        output = multi_num_points_multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights,
            self.num_points)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity

    def _expand_to_total_num_points(self, data: Tensor) -> Tensor:
        assert data.dim() >= 2

        if data.size(-2) == 1:
            return data.expand(*data.shape[:-2], self.total_num_points,
                               *data.shape[-1:])

        ref_list = []
        for i, n in enumerate(self.num_points):
            ref_list.append(
                data.select(-2,
                            i).unsqueeze(-2).expand(*data.shape[:-2], n,
                                                    *data.shape[-1:]))
        return torch.cat(ref_list, dim=-2)


class DFINETransformerDecoderLayer(DeformableDetrTransformerDecoderLayer):
    """Decoder layer of D-FINE."""

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)

        num_points = self.cross_attn_cfg.get('num_points', None)
        if num_points is None or isinstance(num_points, int):
            self.cross_attn = MultiScaleDeformableAttention(
                **self.cross_attn_cfg)
        else:
            self.cross_attn = MultiNumPointsMultiScaleDeformableAttention(
                **self.cross_attn_cfg)
        self.cross_attn.value_proj = nn.Identity()
        self.cross_attn.output_proj = nn.Identity()

        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)
        self.gateway = Gate(self.embed_dims)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                **kwargs) -> Tensor:
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """

        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        query = self.norms[0](query)

        assert kwargs.pop('identity', None) is None
        query_ = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            identity=0,
            **kwargs)
        query = self.gateway(query, query_)

        query = self.norms[1](query)
        query = self.ffn(query)

        query = query.clamp(min=-65504, max=65504)
        query = self.norms[2](query)

        return query


class LQE(nn.Module):

    def __init__(self, k: int, hidden_dim: int, num_layers: int, reg_max: int):
        super(LQE, self).__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers)
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.constant_(self.reg_conf.layers[-1].bias, 0)
        nn.init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores: Tensor, pred_corners: Tensor) -> Tensor:
        B, L, _ = pred_corners.size()
        prob = F.softmax(
            pred_corners.reshape(B, L, 4, self.reg_max + 1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat(
            [prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: ``sum{P(y_i) * y_i}``,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Defaults to 16.
            You may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max: int = 16, reg_scale: float = 4.0) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             weighting_function(self.reg_max, 0.5, reg_scale))

    def forward(self, x: Tensor) -> Tensor:
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(*shape[:-1], -1)
        return x


class DFINETransformerDecoder(RTDETRTransformerDecoder):
    """Transformer decoder of D-FINE."""

    def __init__(self,
                 *args,
                 reg_max: int = 32,
                 reg_scale: float = 4,
                 layer_scale: float = 1.0,
                 eval_idx: int = -1,
                 num_layers: int = 6,
                 remove_cross_attn_value_proj_and_output_proj: bool = True,
                 **kwargs) -> None:
        if eval_idx < 0:
            eval_idx = num_layers + eval_idx
            assert eval_idx >= 0
        self.eval_idx = eval_idx
        self.reg_max = reg_max
        self.reg_scale = reg_scale
        self.layer_scale = layer_scale
        self.remove_cross_attn_value_proj_and_output_proj = \
            remove_cross_attn_value_proj_and_output_proj
        super().__init__(*args, num_layers=num_layers, **kwargs)

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        num_wide_layers = self.num_layers - self.eval_idx - 1
        self.layers = ModuleList([
            DFINETransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers - num_wide_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims

        if num_wide_layers > 0:
            wide_layer_cfg = self.layer_cfg.deepcopy()

            scaled_dim = int(round(self.layer_scale * self.embed_dims))
            if scaled_dim != self.embed_dims:
                for key in {'self_attn_cfg', 'cross_attn_cfg', 'ffn_cfg'}:
                    if key not in wide_layer_cfg:
                        import inspect
                        parameters = inspect.signature(
                            DFINETransformerDecoderLayer.__init__).parameters
                        wide_layer_cfg[key] = deepcopy(parameters[key].default)
                    wide_layer_cfg[key]['embed_dims'] = scaled_dim

            self.layers.extend([
                DFINETransformerDecoderLayer(**wide_layer_cfg)
                for _ in range(num_wide_layers)
            ])
        self.scaled_dim = self.layers[-1].embed_dims

        if self.remove_cross_attn_value_proj_and_output_proj:
            for layer in self.layers:
                layer.cross_attn.value_proj = nn.Identity()
                layer.cross_attn.output_proj = nn.Identity()

        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')

        self.ref_point_head = MLP(4, self.embed_dims * 2, self.embed_dims, 2)

        self.integral = Integral(self.reg_max, self.reg_scale)
        self.lqe_layers = ModuleList(
            [LQE(4, 64, 2, self.reg_max) for _ in range(self.num_layers)])

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                cls_branches: nn.ModuleList, **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.
            cls_branches: (obj:`nn.ModuleList`): Used for classification
                results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        assert self.return_intermediate
        assert reg_branches is not None
        assert reference_points.shape[-1] == 4
        # To avoid inverse_sigmoid, remove .sigmoid() in pre_decoder
        # So reference_points is unactivated reference_points
        unact_reference_points = reference_points
        reference_points = unact_reference_points.sigmoid()

        eval_idx = kwargs.pop('eval_idx', -1)
        if eval_idx < 0:
            eval_idx = eval_idx + self.num_layers
            assert eval_idx >= 0
        assert eval_idx == self.eval_idx

        all_layers_outputs_classes = []
        all_layers_outputs_coords = []
        all_layers_outputs_corners = []

        query_detach = 0
        pred_corners_undetach = 0

        assert len(cls_branches) == self.num_layers + 1
        assert len(reg_branches) == self.num_layers + 2
        pre_bbox_head = reg_branches[-1]

        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[:, :, None]
            query_pos = self.ref_point_head(reference_points)
            query_pos = query_pos.clamp(min=-10, max=10)

            # Adjust scale if needed for detachable wider layers
            if lid > self.eval_idx and self.scaled_dim != self.embed_dims:
                if self.scaled_dim != query_pos.size(-1):
                    query_pos = F.interpolate(query_pos, size=self.scaled_dim)
                if self.scaled_dim != query.size(-1):
                    query = F.interpolate(query, size=self.scaled_dim)
                    value = F.interpolate(value, size=self.scaled_dim)
                    query_detach = query.detach()

            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)

            if lid == 0:
                reference_points_initial = \
                    (pre_bbox_head(query) + unact_reference_points).sigmoid()
                reference_points_initial_detach = \
                    reference_points_initial.detach()

                if self.training:
                    all_layers_outputs_classes.append(cls_branches[0](query))
                    all_layers_outputs_coords.append(reference_points_initial)

            # Refine bounding box corners using FDR,
            # integrating previous layer's corrections
            pred_corners = reg_branches[lid](
                query + query_detach) + pred_corners_undetach
            new_reference_points = distance2bbox(
                reference_points_initial_detach,
                self.integral(pred_corners),
                self.reg_scale,
                clamp_wh=True)

            if self.training or lid == eval_idx:
                # Lqe does not affect the performance here.
                scores = self.lqe_layers[lid](cls_branches[lid](query),
                                              pred_corners)
                all_layers_outputs_classes.append(scores)
                all_layers_outputs_coords.append(new_reference_points)
                all_layers_outputs_corners.append(pred_corners)

                if not self.training or lid == self.num_layers - 1:
                    break

            query_detach = query.detach()
            pred_corners_undetach = pred_corners
            reference_points = new_reference_points.detach()

        if self.training:
            all_layers_outputs_coords = (all_layers_outputs_coords,
                                         all_layers_outputs_corners)

        return all_layers_outputs_classes, all_layers_outputs_coords


class DFINECdnQueryGenerator(CdnQueryGenerator):

    def generate_dn_bbox_query(self, gt_bboxes: Tensor,
                               num_groups: int) -> Tensor:
        """Generate noisy bboxes and their query embeddings.

        The strategy for generating noisy bboxes is as follow:

        .. code:: text

            +--------------------+
            |      negative      |
            |    +----------+    |
            |    | positive |    |
            |    |    +-----|----+------------+
            |    |    |     |    |            |
            |    +----+-----+    |            |
            |         |          |            |
            +---------+----------+            |
                      |                       |
                      |        gt bbox        |
                      |                       |
                      |             +---------+----------+
                      |             |         |          |
                      |             |    +----+-----+    |
                      |             |    |    |     |    |
                      +-------------|--- +----+     |    |
                                    |    | positive |    |
                                    |    +----------+    |
                                    |      negative      |
                                    +--------------------+

         The random noise is added to the top-left and down-right point
         positions, hence, normalized (x, y, x, y) format of bboxes are
         required. The noisy bboxes of positive queries have the points
         both within the inner square, while those of negative queries
         have the points both between the inner and outer squares.

        Besides, the length of outer square is twice as long as that of
        the inner square, i.e., self.box_noise_scale * w_or_h / 2.
        NOTE The noise is added to all the bboxes. Moreover, there is still
        unconsidered case when one point is within the positive square and
        the others is between the inner and outer squares.

        Args:
            gt_bboxes (Tensor): The concatenated gt bboxes of all samples
                in the batch, has shape (num_target_total, 4) with the last
                dimension arranged as (cx, cy, w, h) where
                `num_target_total = sum(num_target_list)`.
            num_groups (int): The number of denoising query groups.

        Returns:
            Tensor: The output noisy bboxes, which are embedded by normalized
            (cx, cy, w, h) format bboxes going through inverse_sigmoid, has
            shape (num_noisy_targets, 4) with the last dimension arranged as
            (cx, cy, w, h), where
            `num_noisy_targets = num_target_total * num_groups * 2`.
        """
        assert self.box_noise_scale > 0
        device = gt_bboxes.device

        # expand gt_bboxes as groups
        gt_bboxes_expand = gt_bboxes.repeat(2 * num_groups, 1)  # xyxy

        # obtain index of negative queries in gt_bboxes_expand
        positive_idx = torch.arange(
            len(gt_bboxes), dtype=torch.long, device=device)
        positive_idx = positive_idx.unsqueeze(0).repeat(num_groups, 1)
        positive_idx += 2 * len(gt_bboxes) * torch.arange(
            num_groups, dtype=torch.long, device=device)[:, None]
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(gt_bboxes)

        # determine the sign of each element in the random part of the added
        # noise to be positive or negative randomly.
        rand_sign = torch.randint_like(
            gt_bboxes_expand, low=0, high=2,
            dtype=torch.float32) * 2.0 - 1.0  # [low, high), 1 or -1, randomly

        # calculate the random part of the added noise
        rand_part = torch.rand_like(gt_bboxes_expand)  # [0, 1)
        rand_part[negative_idx] += 1.0  # pos: [0, 1); neg: [1, 2)
        rand_part *= rand_sign  # pos: (-1, 1); neg: (-2, -1] U [1, 2)

        # add noise to the bboxes
        bboxes_whwh = bbox_xyxy_to_cxcywh(gt_bboxes_expand)[:, 2:].repeat(1, 2)
        noisy_bboxes_expand = gt_bboxes_expand + torch.mul(
            rand_part, bboxes_whwh) * self.box_noise_scale / 2  # xyxy
        noisy_bboxes_expand = noisy_bboxes_expand.clamp(min=0.0, max=1.0)
        noisy_bboxes_expand = bbox_xyxy_to_cxcywh(noisy_bboxes_expand)

        noisy_bboxes_expand = noisy_bboxes_expand.abs()  # TODO: to remove

        dn_bbox_query = inverse_sigmoid(noisy_bboxes_expand, eps=1e-3)
        return dn_bbox_query
