_base_ = './deimv2_dinov3_x_8xb4-58e_coco.py'

# We use DINOv3 as backbone, you can download them following the guide
# in [DINOv3](https://github.com/facebookresearch/dinov3).
pretrained = 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'

base_dim = 384
reg_max = 32
reg_scale = 4
switch_assigner_epoch = 28
norm_cfg = dict(type='GN', num_groups=base_dim // 8, requires_grad=True)

model = dict(
    backbone=dict(
        name='dinov3_vitl16',
        weights_path=pretrained,
        interaction_indexes=[11, 17, 23],  # only need the [1/8, 1/16, 1/32]
        # finetune=False,
        conv_inplane=192,
        hidden_dim=base_dim),
    encoder=dict(
        in_channels=[base_dim, base_dim, base_dim],
        fpn_cfg=dict(
            in_channels=[base_dim, base_dim, base_dim],
            out_channels=base_dim,
            norm_cfg=norm_cfg,
            expansion=1.5),
        layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=base_dim, feedforward_channels=base_dim * 4),
            self_attn_cfg=dict(embed_dims=base_dim))),
    decoder=dict(
        reg_max=reg_max,
        reg_scale=reg_scale,
        ref_hidden_dim=base_dim,
        layer_cfg=dict(
            cross_attn_cfg=dict(embed_dims=base_dim),
            ffn_cfg=dict(embed_dims=base_dim, feedforward_channels=base_dim * 4),
            self_attn_cfg=dict(embed_dims=base_dim))),
    bbox_head=dict(
        reg_max=reg_max,
        reg_scale=reg_scale,
        embed_dims=base_dim),
    train_cfg=dict(switch_assigner=dict(switch_epoch=switch_assigner_epoch)))

backbone_lr_mult = 0.005
custom_keys = {
    'in_proj_bias':
    dict(decay_mult=0),
    'backbone.dinov3':
    dict(lr_mult=backbone_lr_mult),
    'backbone.dinov3.norm.weight':
    dict(lr_mult=backbone_lr_mult, decay_mult=0),
    'backbone.dinov3.norm.bias':
    dict(lr_mult=backbone_lr_mult, decay_mult=0),
    'backbone.dinov3.patch_embed.proj.bias':
    dict(lr_mult=backbone_lr_mult, decay_mult=0),
    # TODO the following norm layers' weight will apply weight decay
    # 'backbone.norms': dict(decay_mult=1),
    # 'backbone.sta.stem.1.weight': dict(decay_mult=1),
    # 'backbone.sta.conv2.1.weight': dict(decay_mult=1),
    # 'backbone.sta.conv3.2.weight': dict(decay_mult=1),
    # 'backbone.sta.conv4.2.weight': dict(decay_mult=1),
}
custom_keys.update({
    f'backbone.dinov3.blocks.{bid}.{name}':
    dict(lr_mult=backbone_lr_mult, decay_mult=0)
    for name in [
        'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias',
        'attn.qkv.bias', 'attn.qkv.bias_mask', 'attn.proj.bias',
        'mlp.fc1.bias', 'mlp.fc2.bias'
    ] for bid in range(12)
})
custom_keys.update({
    f'decoder.layers.{lid}.norms.{i}.scale': dict(decay_mult=0)
    for lid in range(_base_.num_layers) for i in range(3)
})

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.00025),
    paramwise_cfg=dict(custom_keys=dict(_delete_=True, **custom_keys)))

# learning policy
max_epochs = 36
train_cfg = dict(max_epochs=max_epochs)

stage2_switch_epoch = 4
stage3_switch_epoch = 18
stage4_switch_epoch = 32
custom_hooks = [
    dict(type='SetEpochInfoHook'),  # for DEIMV2 assigner switch
    dict(
        type='EMADynamicMomentumHook',
        restart_epoch=stage4_switch_epoch,
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        gamma=1000,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=stage2_switch_epoch,
        switch_pipeline=_base_.train_pipeline_stage2),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=stage3_switch_epoch,
        switch_pipeline=_base_.train_pipeline_stage3),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=stage4_switch_epoch,
        switch_pipeline=_base_.train_pipeline_stage4),
    dict(
        type='DataPreprocessorSwitchHook',
        switch_epoch=stage2_switch_epoch,
        switch_data_preprocessor=_base_.data_preprocessor_stage2),
    dict(
        type='DataPreprocessorSwitchHook',
        switch_epoch=stage3_switch_epoch,
        switch_data_preprocessor=_base_.data_preprocessor_stage3),
    dict(
        type='DataPreprocessorSwitchHook',
        switch_epoch=stage4_switch_epoch,
        switch_data_preprocessor=_base_.data_preprocessor_stage4)
]

param_scheduler = [
    dict(type='QuadraticWarmupLR', by_epoch=False, begin=0, end=2000),
    dict(
        type='CosineAnnealingLR',
        begin=stage3_switch_epoch,
        end=stage4_switch_epoch,
        by_epoch=True,
        eta_min_ratio=0.5,
        convert_to_iter_based=True),
    dict(
        type='ConstantLR', by_epoch=True, factor=1, begin=stage4_switch_epoch)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(enable=False, base_batch_size=16)
