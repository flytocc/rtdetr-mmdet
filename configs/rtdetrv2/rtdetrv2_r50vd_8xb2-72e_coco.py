_base_ = '../rtdetr/rtdetr_r50vd_8xb2-72e_coco.py'

model = dict(
    data_preprocessor=dict(batch_augments=[
        dict(
            type='BatchSyncRandomResize',
            interval=1,
            interpolations='nearest',
            random_sizes=[480, 512, 544, 576, 608] +
            [640] * _base_.base_size_repeat + [672, 704, 736, 768, 800])
    ]))

data_preprocessor_stage2 = dict(
    type='DetDataPreprocessor',
    mean=[0, 0, 0],
    std=[255, 255, 255],
    bgr_to_rgb=True,
    pad_size_divisor=1)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
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
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    dataset=dict(filter_cfg=None, pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

stage2_switch_epoch = 71
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49),
    dict(
        type='DataPreprocessorSwitchHook',
        switch_epoch=stage2_switch_epoch,
        switch_data_preprocessor=data_preprocessor_stage2),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=stage2_switch_epoch,
        switch_pipeline=train_pipeline_stage2)
]
