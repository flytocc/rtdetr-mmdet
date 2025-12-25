_base_ = './rtdetr-ins_r50vd_8xb2-72e_coco.py'

num_prototypes = 64
mask_dims = 32

model = dict(
    type='RTDETRInsPlus',
    mask_dims=mask_dims,
    num_prototypes=num_prototypes,
    backbone=dict(out_indices=(0, 1, 2, 3)),
    neck=[
        dict(
            type='ChannelMapper',
            in_channels=[256, 512, 1024, 2048],
            kernel_size=1,
            out_channels=[None, 256, 256, 256],
            act_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True)),
        dict(
            type='ChannelMapper',
            in_channels=[256, 256, 256, 256],
            kernel_size=3,
            out_channels=[64, None, None, None],
            act_cfg=dict(type='SiLU', inplace=True),
            norm_cfg=dict(type='BN', requires_grad=True)),
    ],
    bbox_head=dict(
        mask_dims=mask_dims,
        loss_cls=dict(loss_weight=2.0)),  # 1.0 in RTDETR
    mask_feat_cfg=dict(num_prototypes=num_prototypes),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='FocalLossCost', weight=4.0),  # 2.0 in RTDETR
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0),
                dict(
                    type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dict(type='DiceCost', weight=5.0, pred_act=True, eps=1.0)
            ])))
