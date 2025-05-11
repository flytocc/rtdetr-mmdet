_base_ = '../rtdetrv2/rtdetrv2_r50vd_8xb2-72e_coco.py'

base_dim = 256
num_points = [3, 6, 3]
reg_max = 32
reg_scale = 4
layer_scale = 1.0
eval_idx = -1
base_size_repeat = 6

model = dict(
    type='DFINE',
    eval_idx=eval_idx,
    data_preprocessor=dict(batch_augments=[
        dict(
            type='BatchSyncRandomResize',
            interval=1,
            interpolations='nearest',
            random_sizes=[480, 512, 544, 576, 608] + [640] * base_size_repeat +
            [672, 704, 736, 768, 800])
    ]),
    backbone=dict(
        _delete_=True,
        type='HGNetV2',
        name='B2',
        return_idx=[1, 2, 3],
        freeze_at=-1,
        freeze_norm=False,
        use_lab=True,
        local_model_dir='/home/nieyang/.cache/torch/hub/checkpoints/'),
    neck=dict(in_channels=[384, 768, 1536]),
    encoder=dict(fpn_cfg=dict(type='DFINEFPN', num_csp_blocks=2)),
    decoder=dict(
        reg_max=reg_max,
        reg_scale=reg_scale,
        layer_scale=layer_scale,
        eval_idx=eval_idx,
        num_layers=4,
        layer_cfg=dict(cross_attn_cfg=dict(num_points=num_points))),
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
    dict(type='PhotoMetricDistortion', hue_delta=12.75),
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

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

# learning policy
max_epochs = 132
train_cfg = dict(max_epochs=max_epochs)

# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={'in_proj_bias': dict(decay_mult=0)}, bias_decay_mult=0))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000)
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
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_data_preprocessor=_base_.data_preprocessor_stage2),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=_base_.train_pipeline_stage2)
]
