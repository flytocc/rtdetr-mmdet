_base_ = './deimv2_hgnetv2_pico_8xb16-500e_coco.py'

base_dim = 96

model = dict(
    num_queries=150,
    backbone=dict(name='Femto'),
    neck=dict(out_channels=base_dim),
    encoder=dict(
        in_channels=[base_dim, base_dim],
        fpn_cfg=dict(in_channels=[base_dim, base_dim], out_channels=base_dim)),
    decoder=dict(
        ref_hidden_dim=base_dim,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=base_dim),
            cross_attn_cfg=dict(embed_dims=base_dim),
            ffn_cfg=dict(
                embed_dims=base_dim,
                # the implementation is different from official DEIMV2 repo
                # `feedforward_channels` shuold be half of that in official
                feedforward_channels=128))),  # SwiGLUFFN
    bbox_head=dict(embed_dims=base_dim))

train_pipeline = [
    dict(type='FilterAnnotations', min_gt_bbox_wh=(10, 10), keep_empty=False),
    dict(type='Resize', scale=(416, 416), keep_ratio=False),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(10, 10), keep_empty=False),
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
                    min_gt_bbox_wh=(10, 10),
                    keep_empty=False),
                dict(type='Resize', scale=(416, 416), keep_ratio=False)
            ],
            [
                dict(
                    type='Mosaic',
                    img_scale=(208, 208),
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
    dict(type='FilterAnnotations', min_gt_bbox_wh=(10, 10), keep_empty=False),
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

backend = 'pillow'  # official impl
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args={{_base_.backend_args}},
        imdecode_backend=backend),
    dict(type='Resize', scale=(416, 416), keep_ratio=False, backend=backend),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

custom_hooks = [
    dict(type='SetEpochInfoHook'),  # for DEIMV2 assigner switch
    dict(
        type='EMADynamicMomentumHook',
        restart_epoch=_base_.stage4_switch_epoch,
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        gamma=1000,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=_base_.stage2_switch_epoch,
        switch_pipeline=train_pipeline_stage2),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=_base_.stage3_switch_epoch,
        switch_pipeline=train_pipeline_stage3),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=_base_.stage4_switch_epoch,
        switch_pipeline=train_pipeline_stage4)
]
