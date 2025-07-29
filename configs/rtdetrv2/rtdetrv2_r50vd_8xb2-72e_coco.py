_base_ = '../rtdetr/rtdetr_r50vd_8xb2-72e_coco.py'

# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={'in_proj_bias': dict(decay_mult=1.0)}, bias_decay_mult=1.0))

data_preprocessor_stage2 = dict(
    type='DetDataPreprocessor',
    mean=[0, 0, 0],
    std=[255, 255, 255],
    bgr_to_rgb=True,
    pad_size_divisor=1)

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

stage2_num_epochs = 1
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49),
    dict(
        type='DataPreprocessorSwitchHook',
        switch_epoch=_base_.max_epochs - stage2_num_epochs,
        switch_data_preprocessor=data_preprocessor_stage2),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=_base_.max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]
