norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='IterativeDecodeHead',
        num_stages=3,
        kernel_update_head=[
            dict(
                type='KernelUpdateHead',
                num_classes=150,
                num_ffn_fcs=2,
                num_heads=8,
                num_mask_fcs=1,
                feedforward_channels=2048,
                in_channels=512,
                out_channels=512,
                dropout=0.0,
                conv_kernel_size=1,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(
                    conv_cfg=dict(type='Conv2d'), act_cfg=None),
                kernel_updator_cfg=dict(
                    type='KernelUpdator',
                    in_channels=256,
                    feat_channels=256,
                    out_channels=256,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN'))) for _ in range(3)
        ],
        kernel_generate_head=dict(
            type='ASPPHead',
            in_channels=2048,
            in_index=3,
            channels=512,
            dilations=(1, 12, 24, 36),
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=[
                dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0),
                dict(type='DiceLoss', loss_name='loss_dice', loss_weight=2.0)])))


train_cfg = dict()
test_cfg = dict(mode='whole')
dataset_type = 'MyDataset'
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(300, 300)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize',  mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data_root = "./aluminum_alloy_samples/"
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/images',
        ann_dir='train/labels',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/images',
        ann_dir='test/labels',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/images',
        ann_dir='test/labels',
        pipeline=test_pipeline))
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook', by_epoch=False)
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = False
find_unused_parameters=True
optimizer = dict(type='Adam', lr=5e-6, betas=(0.9, 0.999))
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
runner = dict(
    type='IterBasedRunner',
    max_iters=100000)
checkpoint_config = dict(
    by_epoch=False,
    save_optimizer=False,
    interval=100000)
evaluation = dict(
    interval=5000,
    metric=['mIoU', 'mFscore', 'mDice'])
