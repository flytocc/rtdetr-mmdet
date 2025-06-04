_base_ = './dfine_hgnetv2_m_8xb4-132e_coco.py'

pretrained = 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'  # noqa

base_size_repeat = 20

model = dict(
    data_preprocessor=dict(batch_augments=[
        dict(
            type='BatchSyncRandomResize',
            interval=1,
            interpolations='nearest',
            random_sizes=[480, 512, 544, 576, 608] + [640] * base_size_repeat +
            [672, 704, 736, 768, 800])
    ]),
    backbone=dict(name='B0', init_cfg=dict(checkpoint=pretrained)),
    neck=dict(in_channels=[256, 512, 1024]),
    encoder=dict(fpn_cfg=dict(num_csp_blocks=1, expansion=0.5)),
    decoder=dict(num_layers=3))

# set all norm layers in backbone to lr_mult=0.5 and decay_mult=0.0
# set all other layers in backbone to lr_mult=0.5
num_blocks_list = (1, 1, 2, 1)
backbone_norm_multi = dict(lr_mult=0.5, decay_mult=1.0)  # NOTE decay_mult=0 ?
custom_keys = {
    'backbone': dict(lr_mult=0.5), 'in_proj_bias': dict(decay_mult=0)}
custom_keys.update({
    f'backbone.stem.{name}.bn': backbone_norm_multi
    for name in ['stem1', 'stem2a', 'stem2b', 'stem3', 'stem4']
})
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.layers.{lid}.bn':
    backbone_norm_multi
    for stage_id, num_blocks in enumerate((1, 1))
    for block_id in range(num_blocks)
    for lid in range(3)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.layers.{lid}.conv{cid}.bn':
    backbone_norm_multi
    for stage_id, num_blocks in enumerate(num_blocks_list[2:], start=2)
    for block_id in range(num_blocks)
    for lid in range(3)
    for cid in (1, 2)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.aggregation.{lid}.bn':
    backbone_norm_multi
    for stage_id, num_blocks in enumerate(num_blocks_list)
    for block_id in range(num_blocks)
    for lid in range(2)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.bn': backbone_norm_multi
    for stage_id in range(1, 4)
})

# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys=dict(_delete_=True, **custom_keys)))
