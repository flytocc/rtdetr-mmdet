_base_ = './rtdetrv2_r50vd_8xb2-72e_coco.py'
pretrained = 'https://github.com/flytocc/mmdetection/releases/download/model_zoo/resnet101vd_ssld_pretrained_64ed664a.pth'  # noqa

model = dict(
    backbone=dict(
        depth=101, init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(out_channels=384),
    encoder=dict(
        in_channels=[384, 384, 384],
        fpn_cfg=dict(in_channels=[384, 384, 384]),
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=384),
            ffn_cfg=dict(embed_dims=384, feedforward_channels=2048))))

# set all norm layers in backbone to decay_multi=0.0
# set all other layers in backbone to lr_mult=0.01
num_blocks_list = (3, 4, 23, 3)  # r101
downsample_norm_idx_list = (3, 3, 3, 3)  # r101
backbone_norm_multi = dict(decay_mult=0.0)
custom_keys = {'backbone': dict(lr_mult=0.01)}
custom_keys.update({
    'backbone.stem.1': backbone_norm_multi,
    'backbone.stem.4': backbone_norm_multi,
    'backbone.stem.7': backbone_norm_multi,
})
custom_keys.update({
    f'backbone.layer{stage_id + 1}.{block_id}.bn': backbone_norm_multi
    for stage_id, num_blocks in enumerate(num_blocks_list)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.layer{stage_id + 1}.{block_id}.downsample.{downsample_norm_idx - 1}':  # noqa
    backbone_norm_multi
    for stage_id, (num_blocks, downsample_norm_idx) in enumerate(
        zip(num_blocks_list, downsample_norm_idx_list))
    for block_id in range(num_blocks)
})

# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(_delete_=True, **custom_keys), bias_decay_mult=1.0))
