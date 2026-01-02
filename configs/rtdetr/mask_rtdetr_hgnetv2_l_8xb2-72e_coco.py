_base_ = './rtdetr-ins_r50vd_8xb2-72e_coco.py'

pretrained = 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth'  # noqa

num_prototypes = 64
mask_dims = 32

model = dict(
    type='MaskRTDETR',
    mask_dims=mask_dims,
    num_prototypes=num_prototypes,
    backbone=dict(
        _delete_=True,
        type='HGNetV2',
        name='B4',
        return_idx=[0, 1, 2, 3],
        freeze_at=0,
        freeze_norm=True,
        use_lab=False,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=[
        dict(
            type='ChannelMapper',
            in_channels=[128, 512, 1024, 2048],
            kernel_size=1,
            out_channels=[None, 256, 256, 256],
            act_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True)),
        dict(
            type='ChannelMapper',
            in_channels=[128, 256, 256, 256],
            kernel_size=3,
            out_channels=[64, None, None, None],
            act_cfg=dict(type='SiLU', inplace=True),
            norm_cfg=dict(type='BN', requires_grad=True)),
    ],
    bbox_head=dict(
        type='MaskRTDETRHead_ppdet',
        vfl_iou_type='mask',
        mask_dims=mask_dims,
        loss_cls=dict(loss_weight=4.0)),  # 1.0 in RTDETR
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

# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.05)}))
