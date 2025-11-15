_base_ = '../deim/deim_hgnetv2_m_8xb4-102e_coco.py'

switch_assigner_epoch = 80

model = dict(
    type='DEIMV2',
    encoder=dict(fpn_cfg=dict(fuse_type='sum')),
    decoder=dict(
        ref_hidden_dim=_base_.base_dim,
        ref_num_layers=3,
        update_query_pos=False,
        layer_cfg=dict(
            ffn_cfg=dict(
                _delete_=True,
                embed_dims=_base_.base_dim,
                # the implementation is different from official DEIMV2 repo
                # `feedforward_channels` shuold be half of that in official
                feedforward_channels=_base_.base_dim * 2))),  # SwiGLUFFN
    train_cfg=dict(
        switch_assigner=dict(
            switch_epoch=switch_assigner_epoch,
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(
                        type='DEIMV2LossCost', iou_order_alpha=4.0, weight=1.)
                ]))))

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
                        expand_ratios=(0.1, 0.25),
                        ratio_range=(0.45, 0.55),
                        prob=0.5)
                ],
            ]),
        dict(
            type='BatchSyncRandomResize',
            interval=1,
            interpolations='nearest',
            random_sizes=[480, 512, 544, 576, 608] +
            [640] * _base_.base_size_repeat + [672, 704, 736, 768, 800])
    ],
    mean=[0, 0, 0],
    std=[255, 255, 255],
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
            expand_ratios=(0.1, 0.25),
            ratio_range=(0.45, 0.55),
            prob=0.5),
        dict(
            type='BatchSyncRandomResize',
            interval=1,
            interpolations='nearest',
            random_sizes=[480, 512, 544, 576, 608] +
            [640] * _base_.base_size_repeat + [672, 704, 736, 768, 800])
    ],
    mean=[0, 0, 0],
    std=[255, 255, 255],
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
        switch_data_preprocessor=_base_.data_preprocessor_stage4)
]
