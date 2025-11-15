_base_ = 'rtdetr-ins_r50vd_8xb2-72e_coco.py'

model = dict(
    type='RTDETRInsPlus',
    backbone=dict(out_indices=(0, 1, 2, 3)),
    neck=dict(in_channels=[256, 512, 1024, 2048]))
