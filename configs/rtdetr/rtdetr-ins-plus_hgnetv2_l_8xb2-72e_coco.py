_base_ = './rtdetr-ins-plus_r50vd_8xb2-72e_coco.py'

pretrained = 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True,
        type='HGNetV2',
        name='B4',
        return_idx=[0, 1, 2, 3],
        freeze_at=0,
        freeze_norm=True,
        use_lab=False,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=[
        dict(
            type='ChannelMapper',
            in_channels=[128, 512, 1024, 2048],
            kernel_size=1,
            out_channels=[None, 256, 256, 256],
            act_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True)),
        dict(
            type='ChannelMapper',
            in_channels=[128, 256, 256, 256],
            kernel_size=1,
            out_channels=[64, None, None, None],
            act_cfg=dict(type='SiLU', inplace=True),
            norm_cfg=dict(type='BN', requires_grad=True)),
    ])

# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.05)}))
