_base_ = './rtdetrv2_r50vd_8xb2-72e_coco.py'
pretrained = 'https://github.com/flytocc/mmdetection/releases/download/model_zoo/resnet18vd_pretrained_265f1124.pth'  # noqa

model = dict(
    data_preprocessor=dict(batch_augments=None),
    backbone=dict(
        depth=18,
        frozen_stages=-1,
        norm_cfg=dict(requires_grad=True),
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[128, 256, 512]),
    encoder=dict(fpn_cfg=dict(expansion=0.5)),
    decoder=dict(num_layers=3))

# optimizer
optim_wrapper = dict(paramwise_cfg=dict(custom_keys=dict(_delete_=True)))

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
