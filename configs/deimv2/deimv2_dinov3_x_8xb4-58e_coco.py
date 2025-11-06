_base_ = '../deim/deim_hgnetv2_l_8xb4-58e_coco.py'

pretrained = 'dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'

base_size_repeat = 3

num_layers = _base_.model.decoder.num_layers  # 6

model = dict(
    type='DEIMV2',
    data_preprocessor=dict(
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                interval=1,
                interpolations='nearest',
                random_sizes=[480, 512, 544, 576, 608] +
                [640] * base_size_repeat + [672, 704, 736, 768, 800])
        ],
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]),
    backbone=dict(
        _delete_=True,
        type='DINOv3STAs',
        name='dinov3_vits16plus',
        weights_path=pretrained,
        interaction_indexes=[5, 8, 11],  # only need the [1/8, 1/16, 1/32]
        finetune=True,
        conv_inplane=64,
        hidden_dim=256),
    neck=None,
    encoder=dict(
        fpn_cfg=dict(
            type='DEIMV2FPN',
            fuse_type='sum',
            expansion=1.25,
            num_csp_blocks=4)),
    decoder=dict(
        ref_hidden_dim=256,
        ref_num_layers=3,
        layer_cfg=dict(
            ffn_cfg=dict(
                _delete_=True, embed_dims=256,
                # the implementation is different from official DEIMV2 repo
                # `feedforward_channels` shuold be half of that in official
                feedforward_channels=1024))),  # SwiGLUFFN
    train_cfg=dict(
        switch_assigner=dict(
            switch_epoch=45,
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(
                        type='DEIMV2LossCost', iou_order_alpha=4.0, weight=1.)
                ]))))

custom_keys = {
    'in_proj_bias': dict(decay_mult=0),
    'backbone.dinov3': dict(lr_mult=0.02),
    'backbone.dinov3.norm.weight': dict(lr_mult=0.02, decay_mult=0),
    'backbone.dinov3.norm.bias': dict(lr_mult=0.02, decay_mult=0),
    'backbone.dinov3.patch_embed.proj.bias': dict(lr_mult=0.02, decay_mult=0),
    # TODO the following norm layers' weight will apply weight decay
    # 'backbone.norms': dict(decay_mult=1),
    # 'backbone.sta.stem.1.weight': dict(decay_mult=1),
    # 'backbone.sta.conv2.1.weight': dict(decay_mult=1),
    # 'backbone.sta.conv3.2.weight': dict(decay_mult=1),
    # 'backbone.sta.conv4.2.weight': dict(decay_mult=1),
}
custom_keys.update({
    f'backbone.dinov3.blocks.{bid}.{name}': dict(lr_mult=0.02, decay_mult=0)
    for name in [
        'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias',
        'attn.qkv.bias', 'attn.qkv.bias_mask', 'attn.proj.bias',
        'mlp.fc1.bias', 'mlp.fc2.bias'
    ] for bid in range(12)
})
custom_keys.update({
    f'decoder.layers.{lid}.norms.{i}.scale': dict(decay_mult=0)
    for lid in range(num_layers) for i in range(3)
})

# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(_delete_=True, **custom_keys), bias_decay_mult=0))

data_preprocessor_stage2 = dict(
    type='DetDataPreprocessor',
    batch_augments=[
        dict(
            type='BatchRandomChoice',
            transforms=[
                [dict(type='BatchMixup', ratio_range=(0.45, 0.55))],
                [
                    dict(
                        type='BatchCopyBlend',
                        area_threshold=100,
                        num_objects=3,
                        with_expand=True,
                        expand_ratios=[0.1, 0.25],
                        ratio_range=(0.45, 0.55),
                        prob=0.5)
                ],
            ]),
        dict(
            type='BatchSyncRandomResize',
            interval=1,
            interpolations='nearest',
            random_sizes=[480, 512, 544, 576, 608] + [640] * base_size_repeat +
            [672, 704, 736, 768, 800])
    ],
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=1)
data_preprocessor_stage3 = dict(
    type='DetDataPreprocessor',
    batch_augments=[
        dict(
            type='BatchCopyBlend',
            area_threshold=100,
            num_objects=3,
            with_expand=True,
            expand_ratios=[0.1, 0.25],
            ratio_range=(0.45, 0.55),
            prob=0.5),
        dict(
            type='BatchSyncRandomResize',
            interval=1,
            interpolations='nearest',
            random_sizes=[480, 512, 544, 576, 608] + [640] * base_size_repeat +
            [672, 704, 736, 768, 800])
    ],
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=1)
data_preprocessor_stage4 = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=1)

custom_hooks = [
    dict(type='SetEpochInfoHook'),  # for DEIMV2 assigner switch
    dict(
        type='EMADynamicMomentumHook',
        restart_epoch=_base_.stage4_switch_epoch,
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        gamma=1000,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=_base_.stage2_switch_epoch,
        switch_pipeline=_base_.train_pipeline_stage2),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=_base_.stage3_switch_epoch,
        switch_pipeline=_base_.train_pipeline_stage3),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=_base_.stage4_switch_epoch,
        switch_pipeline=_base_.train_pipeline_stage4),
    dict(
        type='DataPreprocessorSwitchHook',
        switch_epoch=_base_.stage2_switch_epoch,
        switch_data_preprocessor=data_preprocessor_stage2),
    dict(
        type='DataPreprocessorSwitchHook',
        switch_epoch=_base_.stage3_switch_epoch,
        switch_data_preprocessor=data_preprocessor_stage3),
    dict(
        type='DataPreprocessorSwitchHook',
        switch_epoch=_base_.stage4_switch_epoch,
        switch_data_preprocessor=data_preprocessor_stage4)
]
