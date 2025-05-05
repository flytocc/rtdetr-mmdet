_base_ = './rtdetrv2_r18vd_8xb2-120e_coco.py'

pretrained = 'rtdetrv2_r18vd_8xb2-120e_coco.pth'  # TODO
model = dict(
    type='RTDETRV2',
    init_cfg=dict(type='Pretrained', checkpoint=pretrained))

# learning policy
max_epochs = 36
train_cfg = dict(max_epochs=max_epochs)

stage2_num_epochs = 4
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
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
