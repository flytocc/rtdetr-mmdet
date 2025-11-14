_base_ = './deimv2_hgnetv2_n_8xb16-160e_coco.py'

base_dim = 112
num_points = (4, 2)
switch_assigner_epoch = 450

model = dict(
    num_queries=200,
    backbone=dict(name='Pico', return_idx=[2]),
    neck=dict(
        type='DEIMV2ChannelMapper',
        in_channels=[512],
        out_channels=base_dim,
        extra_act_cfg=dict(type='SiLU', inplace=True),
        num_outs=2),
    encoder=dict(
        in_channels=[base_dim, base_dim],
        use_encoder_idx=[],
        fpn_cfg=dict(
            type='DEIMV2LiteFPN',
            in_channels=[base_dim, base_dim],
            out_channels=base_dim)),
    decoder=dict(
        ref_hidden_dim=base_dim,
        layer_cfg=dict(
            use_gateway=False,
            self_attn_cfg=dict(embed_dims=base_dim),
            cross_attn_cfg=dict(embed_dims=base_dim, num_points=num_points),
            ffn_cfg=dict(
                embed_dims=base_dim,
                # the implementation is different from official DEIMV2 repo
                # `feedforward_channels` shuold be half of that in official
                feedforward_channels=160))),  # SwiGLUFFN
    bbox_head=dict(
        type='DEIMV2Head',
        share_reg_layer=True,
        embed_dims=base_dim,
        use_uni_set=False,
        fgl_loss_weight=None,
        loss_ld=None),
    train_cfg=dict(switch_assigner=dict(switch_epoch=switch_assigner_epoch)))

# optimizer
optim_wrapper = dict(optimizer=dict(lr=0.0016))

# learning policy
max_epochs = 500
train_cfg = dict(max_epochs=max_epochs)

train_pipeline = [
    dict(type='FilterAnnotations', min_gt_bbox_wh=(8, 8), keep_empty=False),
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(8, 8), keep_empty=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
train_pipeline_stage2 = [
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='PhotoMetricDistortion',
                    hue_delta=12.75,
                    clip_val=255,
                    force_float32=False),
                dict(type='Expand', mean=[0, 0, 0]),
                dict(
                    type='RandomApply',
                    transforms=dict(
                        type='MinIoURandomCrop',
                        cover_all_box=False,
                        trials=40),
                    prob=0.8),
                dict(
                    type='FilterAnnotations',
                    min_gt_bbox_wh=(8, 8),
                    keep_empty=False),
                dict(type='Resize', scale=(640, 640), keep_ratio=False)
            ],
            [
                dict(
                    type='Mosaic',
                    img_scale=(320, 320),
                    center_ratio_range=(1.0, 1.0),
                    pad_val=0),
                dict(
                    type='RandomAffine',
                    scaling_ratio_range=(0.5, 1.5),
                    max_shear_degree=0,
                    border_val=(0, 0, 0),
                    center=None),
                dict(
                    type='PhotoMetricDistortion',
                    hue_delta=12.75,
                    clip_val=255,
                    force_float32=False)
            ],
        ]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(8, 8), keep_empty=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
train_pipeline_stage3 = [
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
    *train_pipeline,
]
train_pipeline_stage4 = train_pipeline

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

stage2_switch_epoch = 4
stage3_switch_epoch = 250
stage4_switch_epoch = 468
custom_hooks = [
    dict(type='SetEpochInfoHook'),  # for DEIMV2 assigner switch
    dict(
        type='EMADynamicMomentumHook',
        restart_epoch=stage4_switch_epoch,
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        gamma=1000,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=stage2_switch_epoch,
        switch_pipeline=train_pipeline_stage2),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=stage3_switch_epoch,
        switch_pipeline=train_pipeline_stage3),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=stage4_switch_epoch,
        switch_pipeline=train_pipeline_stage4)
]

param_scheduler = [
    dict(type='QuadraticWarmupLR', by_epoch=False, begin=0, end=4000),
    dict(
        type='CosineAnnealingLR',
        begin=stage3_switch_epoch,
        end=stage4_switch_epoch,
        by_epoch=True,
        eta_min_ratio=0.5,
        convert_to_iter_based=True),
    dict(
        type='ConstantLR', by_epoch=True, factor=1, begin=stage4_switch_epoch)
]
