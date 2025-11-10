_base_ = './rtdetrv2_r50vd_8xb2-72e_coco.py'
pretrained = 'https://github.com/flytocc/mmdetection/releases/download/model_zoo/resnet34vd_pretrained_c7f7c84e.pth'  # noqa

model = dict(
    backbone=dict(
        depth=34,
        frozen_stages=-1,
        norm_cfg=dict(requires_grad=True),
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[128, 256, 512]),
    encoder=dict(fpn_cfg=dict(expansion=0.5)),
    decoder=dict(num_layers=4))

# set all norm layers in backbone to lr_mult=0.5 and decay_mult=0.0
# set all other layers in backbone to lr_mult=0.5
num_blocks_list = (3, 4, 6, 3)  # r34
downsample_norm_idx_list = (2, 3, 3, 3)  # r34
backbone_norm_multi = dict(lr_mult=0.5, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.5), 'in_proj_bias': dict(decay_mult=0)}
custom_keys.update({
    'backbone.stem.1': backbone_norm_multi,
    'backbone.stem.4': backbone_norm_multi,
    'backbone.stem.7': backbone_norm_multi,
})
custom_keys.update({
    f'backbone.layer{stage_id + 1}.{block_id}.bn': backbone_norm_multi
    for stage_id, num_blocks in enumerate(num_blocks_list)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.layer{stage_id + 1}.{block_id}.downsample.{downsample_norm_idx - 1}':  # noqa
    backbone_norm_multi
    for stage_id, (num_blocks, downsample_norm_idx) in enumerate(
        zip(num_blocks_list, downsample_norm_idx_list))
    for block_id in range(num_blocks)
})

# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(_delete_=True, **custom_keys), bias_decay_mult=0))

# learning policy
max_epochs = 120
train_cfg = dict(max_epochs=max_epochs)

stage2_switch_epoch = 117
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
        switch_data_preprocessor=_base_.data_preprocessor_stage2),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=stage2_switch_epoch,
        switch_pipeline=_base_.train_pipeline_stage2)
]
