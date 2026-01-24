_base_ = './deimv2_dinov3_x_8xb4-58e_coco.py'

# We use DINOv3 as backbone, you can download them following the guide
# in [DINOv3](https://github.com/facebookresearch/dinov3).
pretrained = 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'

switch_assigner_epoch = 37

model = dict(
    backbone=dict(
        name='dinov3_vitb16', weights_path=pretrained, conv_inplane=128),
    train_cfg=dict(switch_assigner=dict(switch_epoch=switch_assigner_epoch)))

backbone_lr_mult = 0.01
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
    paramwise_cfg=dict(custom_keys=dict(_delete_=True, **custom_keys)))

# learning policy
max_epochs = 48
train_cfg = dict(max_epochs=max_epochs)

stage2_switch_epoch = 4
stage3_switch_epoch = 24
stage4_switch_epoch = 42
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
