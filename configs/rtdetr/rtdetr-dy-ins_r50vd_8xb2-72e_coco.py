_base_ = 'rtdetr-ins_r50vd_8xb2-72e_coco.py'

model = dict(
    bbox_head=dict(type='RTDETRInsDyConvHead'),
    mask_feat_cfg=dict(num_prototypes=8))
