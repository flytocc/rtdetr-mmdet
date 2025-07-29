_base_ = './dfine_hgnetv2_l_8xb4-80e_coco.py'

pretrained = 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B5_stage1.pth'  # noqa

base_dim = 384
reg_scale = 8
base_size_repeat = 3

model = dict(
    data_preprocessor=dict(batch_augments=[
        dict(
            type='BatchSyncRandomResize',
            interval=1,
            interpolations='nearest',
            random_sizes=[480, 512, 544, 576, 608] + [640] * base_size_repeat +
            [672, 704, 736, 768, 800])
    ]),
    backbone=dict(name='B5', init_cfg=dict(checkpoint=pretrained)),
    neck=dict(out_channels=base_dim),
    encoder=dict(
        in_channels=[base_dim, base_dim, base_dim],
        fpn_cfg=dict(in_channels=[base_dim, base_dim, base_dim]),
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=base_dim),
            ffn_cfg=dict(embed_dims=base_dim, feedforward_channels=2048))),
    decoder=dict(reg_scale=reg_scale),
    bbox_head=dict(reg_scale=reg_scale))

# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.01)}))

custom_hooks = [
    dict(
        type='EMADynamicMomentumHook',
        restart_epoch=_base_.max_epochs - _base_.stage2_num_epochs,
        restart_momentum=0.0002,
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        gamma=1000,
        update_buffers=True,
        priority=49),
    dict(
        type='DataPreprocessorSwitchHook',
        switch_epoch=_base_.max_epochs - _base_.stage2_num_epochs,
        switch_data_preprocessor=_base_.data_preprocessor_stage2),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=_base_.max_epochs - _base_.stage2_num_epochs,
        switch_pipeline=_base_.train_pipeline_stage2)
]
