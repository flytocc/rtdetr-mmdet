_base_ = './rtdetrv2_r50vd_8xb2-72e_coco.py'

model = dict(
    eval_idx=2,  # use 3th decoder layer to eval
    encoder=dict(fpn_cfg=dict(expansion=0.5)))
