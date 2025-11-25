_base_ = './deimv2-ins_dinov3_x_8xb4-58e_coco.py'

num_prototypes = 8

# mask_dim 153
# = num_prototypes*dyconv_channels
# + (num_dyconvs-2)*dyconv_channels**2
# + dyconv_channels
# + (num_dyconvs-1)*dyconv_channels
# + 1

model = dict(
    bbox_head=dict(
        type='DFINEInsDyConvHead',
        num_prototypes=num_prototypes,
        dyconv_channels=num_prototypes,
        num_dyconvs=3),
    mask_feat_cfg=dict(num_prototypes=num_prototypes))
