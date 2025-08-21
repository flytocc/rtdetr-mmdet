_base_ = '../rtdetrv2/rtdetrv2_r50vd_8xb2-72e_coco.py'

pretrained = 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth'  # noqa

base_dim = 256
num_points = (3, 6, 3)
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
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
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

train_dataloader = dict(batch_size=4, num_workers=4)

# set all norm layers in backbone to lr_mult=0.1 and decay_mult=0.0
# set all other layers in backbone to lr_mult=0.1
num_blocks_list = (1, 1, 3, 1)
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=1.0)  # NOTE decay_mult=0 ?
custom_keys = {
    'backbone': dict(lr_mult=0.1),
    'in_proj_bias': dict(decay_mult=0)
}
custom_keys.update({
    f'backbone.stem.{name}.bn': backbone_norm_multi
    for name in ['stem1', 'stem2a', 'stem2b', 'stem3', 'stem4']
})
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.layers.{lid}.bn':
    backbone_norm_multi
    for stage_id, num_blocks in enumerate((1, 1))
    for block_id in range(num_blocks) for lid in range(4)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.layers.{lid}.conv{cid}.bn':
    backbone_norm_multi
    for stage_id, num_blocks in enumerate(num_blocks_list[2:], start=2)
    for block_id in range(num_blocks) for lid in range(4) for cid in (1, 2)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.aggregation.{lid}.bn':
    backbone_norm_multi
    for stage_id, num_blocks in enumerate(num_blocks_list)
    for block_id in range(num_blocks) for lid in range(2)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.bn': backbone_norm_multi
    for stage_id in range(1, 4)
})

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.0002),
    paramwise_cfg=dict(
        custom_keys=dict(_delete_=True, **custom_keys), bias_decay_mult=0))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.002, by_epoch=False, begin=0, end=500)
]

auto_scale_lr = dict(base_batch_size=32)

# learning policy
max_epochs = 132
train_cfg = dict(max_epochs=max_epochs)

default_hooks = dict(checkpoint=dict(type='CheckpointAfterValHook'))

stage2_num_epochs = 12
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
        switch_data_preprocessor=_base_.data_preprocessor_stage2),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=_base_.train_pipeline_stage2)
]
