_base_ = '../rtdetr/rtdetr_r18vd_8xb2-72e_coco.py'

reg_max = 32
reg_scale = 4
layer_scale = 1.0
eval_idx = -1

model = dict(
    type='DFINE',
    eval_idx=eval_idx,
    data_preprocessor=dict(batch_augments=None),
    encoder=dict(fpn_cfg=dict(type='DFINEFPN')),
    decoder=dict(
        reg_max=reg_max,
        reg_scale=reg_scale,
        layer_scale=layer_scale,
        eval_idx=eval_idx),
    bbox_head=dict(
        type='DFINEHead',
        reg_max=reg_max,
        reg_scale=reg_scale,
        layer_scale=layer_scale,
        eval_idx=eval_idx,
        loss_ld=dict(
            type='KnowledgeDistillationKLDivLoss',
            T=5,
            reduction='none',
            loss_weight=1.5)))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomApply',
        transforms=dict(type='PhotoMetricDistortion', hue_delta=12.8),
        prob=0.5),
    dict(type='Expand', mean=[0, 0, 0]),
    dict(
        type='RandomApply',
        transforms=dict(type='MinIoURandomCrop', cover_all_box=False),
        prob=0.8),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
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

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.5),
            'in_proj_bias': dict(decay_mult=0),
        },
        bias_decay_mult=0))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000)
]

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
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]

stage2_num_epochs = 12
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        gamma=1000,
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
