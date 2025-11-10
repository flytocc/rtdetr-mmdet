_base_ = './rtdetrv2_r50vd_8xb2-72e_coco.py'
pretrained = 'https://github.com/flytocc/mmdetection/releases/download/model_zoo/resnet101vd_ssld_pretrained_99188ec9.pth'  # noqa

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

# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.01)}))
