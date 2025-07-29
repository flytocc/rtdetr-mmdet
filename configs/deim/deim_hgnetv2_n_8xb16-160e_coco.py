_base_ = '../dfine/dfine_hgnetv2_n_8xb16-160e_coco.py'

act_cfg = dict(type='SiLU', inplace=True)
model = dict(
    type='DEIMDFINE',
    decoder=dict(
        ref_act_cfg=act_cfg, layer_cfg=dict(ffn_cfg=dict(act_cfg=act_cfg))),
    bbox_head=dict(
        reg_act_cfg=act_cfg,
        loss_cls=dict(type='DEIMMalLoss', alpha=1.0, gamma=1.5)))

train_pipeline = [
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
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
                    min_gt_bbox_wh=(1, 1),
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
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
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

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='MultiImageMixDataset',
        dataset={
            **_base_.train_dataloader.dataset,
            'pipeline': [
                dict(
                    type='LoadImageFromFile',
                    backend_args={{_base_.backend_args}}),
                dict(type='LoadAnnotations', with_bbox=True),
            ],
        },
        pipeline=train_pipeline,
        deepcopy=False))

data_preprocessor_stage2 = dict(
    type='DetDataPreprocessor',
    batch_augments=[
        dict(type='BatchMixup', ratio_range=(0.45, 0.55), prob=0.5)
    ],
    mean=[0, 0, 0],
    std=[255, 255, 255],
    bgr_to_rgb=True,
    pad_size_divisor=1)
data_preprocessor_stage3 = _base_.model.data_preprocessor
data_preprocessor_stage4 = dict(
    type='DetDataPreprocessor',
    mean=[0, 0, 0],
    std=[255, 255, 255],
    bgr_to_rgb=True,
    pad_size_divisor=1)

stage2_switch_epoch = 4
stage3_switch_epoch = 78
stage4_switch_epoch = 148
custom_hooks = [
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
        switch_pipeline=train_pipeline_stage4),
    dict(
        type='DataPreprocessorSwitchHook',
        switch_epoch=stage2_switch_epoch,
        switch_data_preprocessor=data_preprocessor_stage2),
    dict(
        type='DataPreprocessorSwitchHook',
        switch_epoch=stage3_switch_epoch,
        switch_data_preprocessor=data_preprocessor_stage3),
    dict(
        type='DataPreprocessorSwitchHook',
        switch_epoch=stage4_switch_epoch,
        switch_data_preprocessor=data_preprocessor_stage4)
]

param_scheduler = [
    dict(type='QuadraticWarmupLR', by_epoch=False, begin=0, end=2000),
    # dict(
    #     type='CosineAnnealingLR',
    #     begin=stage3_switch_epoch,
    #     end=stage4_switch_epoch,
    #     by_epoch=True,
    #     eta_min_ratio=0.5,
    #     convert_to_iter_based=True),
    # dict(
    #     type='ConstantLR',
    #     by_epoch=True,
    #     factor=1,
    #     begin=stage4_switch_epoch)
]
