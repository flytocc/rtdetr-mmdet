_base_ = './dfine_hgnetv2_m_8xb2-132e_coco.py'

base_size_repeat = 20

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
    backbone=dict(name='B0'),
    neck=dict(in_channels=[256, 512, 1024]),
    encoder=dict(fpn_cfg=dict(num_csp_blocks=1, expansion=0.5)),
    decoder=dict(num_layers=3))

# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.5)}))
