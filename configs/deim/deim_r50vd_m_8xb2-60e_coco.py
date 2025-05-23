_base_ = './deim_r50vd_8xb2-60e_coco.py'

model = dict(
    encoder=dict(fpn_cfg=dict(expansion=0.5)), decoder=dict(num_layers=3))
