_base_ = './rtdetrv2_r18vd_8xb2-120e_coco.py'

pretrained = 'https://github.com/flytocc/rtdetr-mmdet/releases/download/rtdetrv2/rtdetrv2_r18vd_8xb2-120e_coco_caff108e.pth'  # noqa
model = dict(
    type='RTDETRV2', init_cfg=dict(type='Pretrained', checkpoint=pretrained))

# learning policy
max_epochs = 36
train_cfg = dict(max_epochs=max_epochs)

stage2_switch_epoch = 33
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
