_base_ = './rtdetr-ins-plus_hgnetv2_l_8xb2-72e_coco.py'

model = dict(
    type='MaskRTDETR_ppdet',
    bbox_head=dict(type='MaskRTDETRHead_ppdet', vfl_iou_type='mask'),
    mask_feat_cfg=dict(
        _delete_=True,
        in_channels=256,
        feat_channels=64,
        num_prototypes=_base_.num_prototypes,
        act_cfg=dict(type='ReLU', inplace=True),
        norm_cfg=dict(type='BN', requires_grad=True)))
