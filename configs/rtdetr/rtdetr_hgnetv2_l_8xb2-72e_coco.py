_base_ = './rtdetr_r50vd_8xb2-72e_coco.py'
pretrained = 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True,
        type='HGNetV2',
        name='B4',
        return_idx=[1, 2, 3],
        freeze_at=0,
        freeze_norm=True,
        use_lab=False,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)))

# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.05)}))
