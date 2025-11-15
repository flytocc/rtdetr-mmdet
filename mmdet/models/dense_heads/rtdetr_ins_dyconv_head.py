# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from .rtdetr_ins_head import RTDETRInsHead


@MODELS.register_module()
class RTDETRInsDyConvHead(RTDETRInsHead):
    """RTDETR Head for Instance with Dynamic Convalution.

    Args:
        num_prototypes (int): Number of mask prototype features extracted
            from the mask head. Defaults to 8.
        dyconv_channels (int): Channel of the dynamic conv layers.
            Defaults to 8.
        num_dyconvs (int): Number of the dynamic convolution layers.
            Defaults to 3.
    """

    def __init__(self,
                 *args,
                 num_prototypes: int = 8,
                 dyconv_channels: int = 8,
                 num_dyconvs: int = 3,
                 **kwargs) -> None:
        self.num_prototypes = num_prototypes
        self.dyconv_channels = dyconv_channels
        self.num_dyconvs = num_dyconvs

        # calculate num dynamic parameters
        weight_nums, bias_nums = [], []
        for i in range(self.num_dyconvs):
            if i == 0:
                weight_nums.append(
                    # mask prototype only (without coordinate features)
                    (self.num_prototypes + 0) * self.dyconv_channels)
                bias_nums.append(self.dyconv_channels * 1)
            elif i == self.num_dyconvs - 1:
                weight_nums.append(self.dyconv_channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.dyconv_channels * self.dyconv_channels)
                bias_nums.append(self.dyconv_channels * 1)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        kwargs['mask_dims'] = self.num_gen_params
        super().__init__(*args, **kwargs)

    def parse_dynamic_params(self, flatten_kernels: Tensor) -> tuple:
        """split kernel head prediction to conv weight and bias."""
        n_inst = flatten_kernels.size(0)
        n_layers = len(self.weight_nums)
        params_splits = list(
            torch.split_with_sizes(
                flatten_kernels, self.weight_nums + self.bias_nums, dim=1))
        weight_splits = params_splits[:n_layers]
        bias_splits = params_splits[n_layers:]
        for i in range(n_layers):
            out_channels = self.dyconv_channels if i < n_layers - 1 else 1
            weight_splits[i] = weight_splits[i].reshape(
                n_inst, out_channels, -1)
            bias_splits[i] = bias_splits[i].reshape(n_inst, out_channels, 1)
        return weight_splits, bias_splits

    def mask_single(self, mask_pred: Tensor, mask_feat: Tensor) -> Tensor:
        h, w = mask_feat.size()[-2:]
        num_queries = mask_pred.size(0)

        # weight shape (num_queries, dyconv_channels, num_prototypes)
        # bias shape (num_queries, dyconv_channels, 1)
        weights, biases = self.parse_dynamic_params(mask_pred.float())

        # x shape (1, num_prototypes, h * w)
        x = mask_feat.flatten(-2).unsqueeze(0).float()

        n_layers = len(weights)
        for i, (weight, bias) in enumerate(zip(weights, biases)):
            # x shape (num_queries, dyconv_channels, h * w)
            with torch.cuda.amp.autocast(enabled=False):
                x = weight @ x + bias
            if i < n_layers - 1:
                x = F.relu(x)
        x = x.reshape(num_queries, h, w)
        return x

    def feat_to_mask(self, mask_preds: Tensor, mask_feats: Tensor) -> Tensor:
        masks = []
        for mask_pred, mask_feat in zip(mask_preds, mask_feats):
            masks.append(self.mask_single(mask_pred, mask_feat))
        return torch.stack(masks, dim=0)
