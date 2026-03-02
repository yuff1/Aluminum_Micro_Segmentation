norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoderFull',
    decode_head=dict(
        type='AttU_Net',
        out_channels=64,
        num_classes=2,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=True, loss_name='loss_bce', loss_weight=2.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=2.0)]))


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
    interval=500,
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
optimizer = dict(type='Adam', lr=5e-5, betas=(0.9, 0.999))
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)
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
