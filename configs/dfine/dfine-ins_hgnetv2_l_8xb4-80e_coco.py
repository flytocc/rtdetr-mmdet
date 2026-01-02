_base_ = [
    '../_base_/datasets/coco_instance.py', '../_base_/default_runtime.py'
]
pretrained = 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth'  # noqa

base_dim = 256
mask_dims = base_dim
num_points = (3, 6, 3)
num_layers = 6
reg_max = 32
reg_scale = 4
layer_scale = 1.0
eval_idx = -1
base_size_repeat = 4

model = dict(
    type='DFINEIns',
    eval_idx=eval_idx,
    num_queries=300,  # num_matching_queries, 900 for DINO
    # spatial_shapes=((80, 80), (40, 40), (
    #     20, 20)),  # for strdies (8, 16, 32) with image_size 640x640. # noqa
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                interval=1,
                interpolations='nearest',
                random_sizes=[480, 512, 544, 576, 608] +
                [640] * base_size_repeat + [672, 704, 736, 768, 800])
        ],
        mean=[0, 0, 0],
        std=[255, 255, 255],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='HGNetV2',
        name='B4',
        return_idx=[1, 2, 3],
        freeze_at=0,
        freeze_norm=True,
        use_lab=False,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=base_dim,
        act_cfg=None,
        norm_cfg=dict(type='BN', requires_grad=True)),  # GN for DINO
    encoder=dict(
        use_encoder_idx=[-1],
        num_encoder_layers=1,
        in_channels=[base_dim, base_dim, base_dim],
        fpn_cfg=dict(
            type='DFINEFPN',
            in_channels=[base_dim, base_dim, base_dim],
            out_channels=base_dim,
            expansion=1.0,
            norm_cfg=dict(type='BN', requires_grad=True)),
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=base_dim, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=base_dim,
                feedforward_channels=base_dim * 4,  # 2048 for DINO
                ffn_drop=0.0,
                act_cfg=dict(type='GELU')))),  # ReLU for DINO
    decoder=dict(
        reg_max=reg_max,
        reg_scale=reg_scale,
        layer_scale=layer_scale,
        eval_idx=eval_idx,
        num_layers=num_layers,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=base_dim, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(
                embed_dims=base_dim,
                num_levels=3,  # 4 for DINO
                dropout=0.0,
                num_points=num_points),
            ffn_cfg=dict(
                embed_dims=base_dim,
                feedforward_channels=base_dim * 4,  # 2048 for DINO
                ffn_drop=0.0)),
        post_norm_cfg=None),
    bbox_head=dict(
        type='DFINEInsHead',
        reg_max=reg_max,
        reg_scale=reg_scale,
        layer_scale=layer_scale,
        eval_idx=eval_idx,
        mask_dims=mask_dims,
        num_classes=80,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='RTDETRVarifocalLoss',  # FocalLoss in DINO
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=2.0),  # 1.0 in RTDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        loss_ld=dict(
            type='KnowledgeDistillationKLDivLoss',
            T=5,
            reduction='none',
            loss_weight=1.5),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0)),
    mask_feat_cfg=dict(
        in_channels=base_dim,
        feat_channels=base_dim // 2,
        num_prototypes=mask_dims,
        act_cfg=dict(type='ReLU', inplace=True),
        norm_cfg=dict(
            type='GN', num_groups=base_dim // 8, requires_grad=True)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        num_points=12544,  # TODO: double size of feature map ?
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=4.0),  # 2.0 in RTDETR
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0),
                dict(
                    type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dict(type='DiceCost', weight=5.0, pred_act=True, eps=1.0)
            ])),
    test_cfg=dict(max_per_img=100, mask_thr_binary=0.5))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='PhotoMetricDistortion',
        hue_delta=12.75,
        clip_val=255,
        force_float32=False),
    dict(type='Expand', mean=[0, 0, 0]),
    dict(
        type='RandomApply',
        transforms=dict(
            type='MinIoURandomCrop', cover_all_box=False, trials=40),
        prob=0.8),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    batch_sampler=None,  # fixed img size does not need AspectRatioBatchSampler
    drop_last=True,
    pin_memory=True,
    dataset=dict(filter_cfg=None, pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=4, num_workers=4, dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

val_evaluator = dict(proposal_nums=[100])
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00025, weight_decay=0.000125),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.05)},
        norm_decay_mult=0,
        bypass_duplicate=True))

# learning policy
max_epochs = 80
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

data_preprocessor_stage2 = dict(
    type='DetDataPreprocessor',
    mean=[0, 0, 0],
    std=[255, 255, 255],
    bgr_to_rgb=True,
    pad_size_divisor=1)

stage2_num_epochs = 8
custom_hooks = [
    dict(
        type='EMADynamicMomentumHook',
        restart_epoch=max_epochs - stage2_num_epochs,
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        gamma=1000,
        update_buffers=True,
        priority=49),
    dict(
        type='DataPreprocessorSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_data_preprocessor=data_preprocessor_stage2),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.002, by_epoch=False, begin=0, end=500)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (4 samples per GPU)
auto_scale_lr = dict(enable=False, base_batch_size=32)

# for `EMADynamicMomentumHook`
default_hooks = dict(checkpoint=dict(type='CheckpointAfterValHook'))
