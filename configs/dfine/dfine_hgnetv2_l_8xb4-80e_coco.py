_base_ = './dfine_hgnetv2_m_8xb4-132e_coco.py'

pretrained = 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth'  # noqa

base_size_repeat = 4

model = dict(
    data_preprocessor=dict(batch_augments=[
        dict(
            type='BatchSyncRandomResize',
            interval=1,
            interpolations='nearest',
            random_sizes=[480, 512, 544, 576, 608] + [640] * base_size_repeat +
            [672, 704, 736, 768, 800])
    ]),
    backbone=dict(
        name='B4',
        freeze_at=0,
        freeze_norm=True,
        use_lab=False,
        init_cfg=dict(checkpoint=pretrained)),
    neck=dict(in_channels=[512, 1024, 2048]),
    encoder=dict(fpn_cfg=dict(num_csp_blocks=3)),
    decoder=dict(num_layers=6))

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.00025, weight_decay=0.000125),
    paramwise_cfg=dict(
        custom_keys=dict(_delete_=True, backbone=dict(lr_mult=0.05)),
        bias_decay_mult=1.0))

# learning policy
max_epochs = 80
train_cfg = dict(max_epochs=max_epochs)

stage2_num_epochs = 8
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
