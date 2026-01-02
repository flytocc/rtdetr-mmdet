_base_ = './deimv2-ins_dinov3_x_8xb4-58e_coco.py'

base_dim = _base_.base_dim
mask_dims = 32
num_prototypes = 64

model = dict(
    type='DEIMV2InsPlus',
    mask_dims=mask_dims,
    num_prototypes=num_prototypes,
    backbone=dict(interaction_indexes=[2, 5, 8, 11]),  # [1/4, 1/8, 1/16, 1/32]
    neck=[
        dict(
            type='ChannelMapper',
            in_channels=[base_dim, base_dim, base_dim, base_dim],
            kernel_size=3,
            out_channels=[num_prototypes, None, None, None],
            act_cfg=dict(type='SiLU', inplace=True),
            norm_cfg=dict(type='BN', requires_grad=True)),
    ],
    bbox_head=dict(mask_dims=mask_dims),
    mask_feat_cfg=dict(num_prototypes=num_prototypes))
