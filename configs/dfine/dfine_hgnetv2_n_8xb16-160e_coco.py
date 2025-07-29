_base_ = './dfine_hgnetv2_s_8xb4-132e_coco.py'

base_dim = 128
num_points = 6
num_levels = 2

model = dict(
    data_preprocessor=dict(batch_augments=None),
    backbone=dict(return_idx=[2, 3]),
    neck=dict(
        in_channels=[512, 1024], out_channels=base_dim, num_outs=num_levels),
    encoder=dict(
        in_channels=[base_dim, base_dim],
        fpn_cfg=dict(
            in_channels=[base_dim, base_dim],
            out_channels=base_dim,
            num_csp_blocks=2,
            expansion=0.34),
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=base_dim),
            ffn_cfg=dict(
                embed_dims=base_dim, feedforward_channels=base_dim * 4))),
    decoder=dict(
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=base_dim),
            cross_attn_cfg=dict(
                embed_dims=base_dim,
                num_levels=num_levels,
                num_points=num_points),
            ffn_cfg=dict(
                embed_dims=base_dim, feedforward_channels=base_dim * 4))),
    bbox_head=dict(embed_dims=base_dim))

train_dataloader = dict(batch_size=16, num_workers=8)

# optimizer
optim_wrapper = dict(optimizer=dict(lr=0.0008))

auto_scale_lr = dict(base_batch_size=128)

# learning policy
max_epochs = 160
train_cfg = dict(max_epochs=max_epochs)

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
