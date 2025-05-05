_base_ = './dfine_hgnetv2_m_8xb2-132e_coco.py'

base_dim = 384
reg_scale = 8
base_size_repeat = 3

model = dict(
    data_preprocessor=dict(
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                interval=1,
                interpolations='nearest',
                random_sizes=
                    [480, 512, 544, 576, 608] +
                    [640] * base_size_repeat +
                    [672, 704, 736, 768, 800]
                )
    ]),
    backbone=dict(
        name='B5',
        freeze_stem_only=True,
        freeze_at=0,
        freeze_norm=True,
        use_lab=False),
    neck=dict(in_channels=[512, 1024, 2048], out_channels=base_dim),
    encoder=dict(
        in_channels=[base_dim, base_dim, base_dim],
        fpn_cfg=dict(
            in_channels=[base_dim, base_dim, base_dim],
            out_channels=base_dim,
            num_csp_blocks=3),
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=base_dim),
            ffn_cfg=dict(embed_dims=base_dim, feedforward_channels=2048))),
    decoder=dict(
        num_layers=6,
        reg_scale=reg_scale,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=base_dim),
            cross_attn_cfg=dict(embed_dims=base_dim),
            ffn_cfg=dict(embed_dims=base_dim, feedforward_channels=2048))),
    bbox_head=dict(reg_scale=reg_scale))

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.000125, weight_decay=0.000125),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.01)}))

# learning policy
max_epochs = 80
train_cfg = dict(max_epochs=max_epochs)

stage2_num_epochs = 8
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,  # TODO 0.0002 in stage2
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
