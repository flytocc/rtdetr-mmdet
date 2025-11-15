# Copyright (c) OpenMMLab. All rights reserved.
import math
from copy import deepcopy
from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import BaseModule, ModuleList
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .dfine_layers import (DFINEFPN, LQE, DFINETransformerDecoder,
                           DFINETransformerDecoderLayer, Gate, Integral,
                           MultiNumPointsMultiScaleDeformableAttention,
                           RepNCSPELAN4, distance2bbox)
from .rtdetr_layers import CSPLayer
from .utils import MLP


class RepNCSPELAN5(RepNCSPELAN4):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: Optional[int] = None,
                 expand_ratio: float = 1.0,
                 num_blocks: int = 3,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super(RepNCSPELAN4, self).__init__(init_cfg=init_cfg)
        hidden_channels = hidden_channels or in_channels
        mid_channels = int(out_channels * expand_ratio // 2)
        self.cv1 = ConvModule(
            in_channels,
            hidden_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.cv2 = nn.Sequential(
            CSPLayer(
                hidden_channels // 2,
                mid_channels,
                expand_ratio=1.0,
                num_blocks=num_blocks,
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
                act_cfg=act_cfg))
        self.cv4 = ConvModule(
            hidden_channels + mid_channels * 2,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)


@MODELS.register_module()
class DEIMV2FPN(DFINEFPN):
    """FPN of DEIM v2."""

    csp_block = RepNCSPELAN5


# Modified from
# https://github.com/meituan/YOLOv6/blob/main/yolov6/layers/common.py#L695
class GAP_Fusion(nn.Module):
    """BiFusion Block in PAN."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='SiLU', inplace=True)):
        super().__init__()
        self.cv = ConvModule(
            in_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        # global average pooling
        gap = F.adaptive_avg_pool2d(x, 1)
        x = x + gap
        return self.cv(x)


# Modified from mmdet/models/necks/channel_mapper.py
@MODELS.register_module()
class DEIMV2ChannelMapper(BaseModule):

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        kernel_size: int = 3,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
        extra_act_cfg: OptConfigType = dict(type='SiLU', inplace=True),
        bias: Union[bool, str] = 'auto',
        num_outs: int = None,
        init_cfg: OptMultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.extra_convs = None
        if num_outs is None:
            num_outs = len(in_channels)
        self.convs = nn.ModuleList()
        for in_channel in in_channels:
            self.convs.append(
                ConvModule(
                    in_channel,
                    out_channels,
                    kernel_size,
                    padding=(kernel_size - 1) // 2,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=bias))
        if num_outs > len(in_channels):
            self.extra_convs = nn.ModuleList()
            for _ in range(len(in_channels), num_outs):
                self.extra_convs.append(
                    nn.Sequential(
                        nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                        ConvModule(
                            out_channels,
                            out_channels,
                            1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=extra_act_cfg)))

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """Forward function."""
        assert len(inputs) == len(self.convs)
        outs = [self.convs[i](inputs[i]) for i in range(len(inputs))]
        if self.extra_convs:
            for i in range(len(self.extra_convs)):
                outs.append(self.extra_convs[i](outs[-1]))
        return tuple(outs)


@MODELS.register_module()
class DEIMV2LiteFPN(DFINEFPN):
    """Lite FPN of DEIM v2.

    Args:
        in_channels (List[int], optional): The input channels of the
            feature maps. Defaults to [256, 256].
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
        in_channels: List[int] = [256, 256],
        out_channels: int = 256,
        num_csp_blocks: int = 3,
        expansion: float = 1.0,
        fuse_type: Literal['cat', 'sum'] = 'cat',
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
        super(DFINEFPN, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fuse_type = fuse_type
        inp_scale = 2 if fuse_type == 'cat' else 1

        # top-down fpn
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                GAP_Fusion(
                    in_channels[idx],
                    in_channels[idx - 1],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.top_down_blocks.append(
                self.csp_block(
                    in_channels[idx - 1] * inp_scale,
                    in_channels[idx - 1],
                    hidden_channels=in_channels[idx - 1] * 2,
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
                    nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                    ConvModule(
                        in_channels[idx],
                        in_channels[idx],
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg)))
            self.bottom_up_blocks.append(
                self.csp_block(
                    in_channels[idx] * inp_scale,
                    in_channels[idx + 1],
                    hidden_channels=in_channels[idx] * 2,
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


class RMSNorm(nn.Module):

    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_features))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if hasattr(torch, 'rms_norm'):  # since 2.4
            return torch.rms_norm(x, (self.num_features, ), self.scale,
                                  self.eps)

        output = self._norm(x.float()).type_as(x)
        output = output * self.scale
        return output

    def extra_repr(self) -> str:
        return f'num_features={self.num_features}, eps={self.eps}'


# Modified from
# https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/swiglu_ffn.py#L14-L34 # noqa
class SwiGLUFFN(nn.Module):

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: Optional[int] = None,
        out_features: Optional[int] = None,
        add_identity=True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.add_identity = add_identity
        out_features = out_features or embed_dims
        feedforward_channels = feedforward_channels or embed_dims
        self.w12 = nn.Linear(embed_dims, 2 * feedforward_channels, bias=bias)
        self.w3 = nn.Linear(feedforward_channels, out_features, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.w12.weight)
        nn.init.constant_(self.w12.bias, 0)
        nn.init.xavier_uniform_(self.w3.weight)
        nn.init.constant_(self.w3.bias, 0)

    def forward(self, x, identity=None):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        if identity is None:
            identity = x
        return identity + self.w3(hidden)


class DEIMV2TransformerDecoderLayer(DFINETransformerDecoderLayer):
    """Decoder layer of DEIM v2."""

    def __init__(self,
                 *args,
                 use_gateway: bool = True,
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256, feedforward_channels=512),
                 norm_cfg: OptConfigType = dict(type=RMSNorm, eps=1e-6),
                 **kwargs) -> None:
        self.use_gateway = use_gateway
        super().__init__(*args, ffn_cfg=ffn_cfg, norm_cfg=norm_cfg, **kwargs)

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
        self.ffn = SwiGLUFFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)
        if self.use_gateway:
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

        if self.use_gateway:
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
        else:
            query = self.cross_attn(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_mask=cross_attn_mask,
                key_padding_mask=key_padding_mask,
                **kwargs)

        query = self.norms[1](query)
        query = self.ffn(query)

        query = query.clamp(min=-65504, max=65504)
        query = self.norms[2](query)

        return query


class DEIMV2TransformerDecoder(DFINETransformerDecoder):
    """Transformer decoder of DEIM v2."""

    def _init_layers(self) -> None:
        """Initialize decoder layers.

        NOTE only changes:
            DFINETransformerDecoderLayer -> DEIMV2TransformerDecoderLayer
        """
        num_wide_layers = self.num_layers - self.eval_idx - 1
        self.layers = ModuleList([
            DEIMV2TransformerDecoderLayer(**self.layer_cfg)
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
                            DEIMV2TransformerDecoderLayer.__init__).parameters
                        wide_layer_cfg[key] = deepcopy(parameters[key].default)
                    wide_layer_cfg[key]['embed_dims'] = scaled_dim

            self.layers.extend([
                DEIMV2TransformerDecoderLayer(**wide_layer_cfg)
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

        self.ref_point_head = MLP(
            4,
            self.ref_hidden_dim or self.embed_dims * 2,
            self.embed_dims,
            self.ref_num_layers,
            act_cfg=self.ref_act_cfg)

        self.integral = Integral(self.reg_max, self.reg_scale)
        if self.with_lqe:
            self.lqe_layers = ModuleList([
                LQE(4, 64, 2, self.reg_max, act_cfg=self.lqe_act_cfg)
                for _ in range(self.num_layers)
            ])
