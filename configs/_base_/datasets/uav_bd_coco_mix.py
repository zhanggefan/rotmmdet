dataset_type = 'UAVBD'
data_root = 'data/uav-bd/'
coco_type = 'CocoDataset'
coco_root = 'data/coco/'
img_norm_cfg = dict(mean=[114, 114, 114], std=[255, 255, 255], to_rgb=True)
train_pipeline = [
    dict(
        type='MosaicPipelineR',
        individual_pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RandomCropR', crop_size=(342, 342),
                 allow_negative_crop=True),
            dict(type='ResizeR', img_scale=(342, 342), keep_ratio=True),
            dict(type='RandomFlipR', flip_ratio=0.5),
        ],
        pad_val=114),
    dict(type='ResizeR', img_scale=(684, 684), ratio_range=(0.75, 1.5),
         keep_ratio=True),
    dict(type='RandomCropR', crop_size=(480, 480)),
    dict(
        type='HueSaturationValueJitter',
        hue_ratio=0.015,
        saturation_ratio=0.7,
        value_ratio=0.4),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(352, 352),
        flip=False,
        transforms=[
            dict(type='ResizeR', keep_ratio=True),
            dict(type='RandomFlipR'),
            dict(type='Pad', size_divisor=32),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=6,
    train=dict(
        type='UAVBD_BGMix',
        ann_file=data_root + 'train.txt',
        img_prefix=data_root + 'train',
        bg_ratio=0.5,
        bg_dataset_conf=dict(
            type=coco_type,
            ann_file=coco_root + 'annotations/instances_train2017.json',
            img_prefix=coco_root + 'train2017/'
        ),
        pipeline=train_pipeline),
    val=dict(
        samples_per_gpu=32,
        type=dataset_type,
        ann_file=data_root + 'val.txt',
        img_prefix=data_root + 'val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.txt',
        img_prefix=data_root + 'test',
        pipeline=test_pipeline))
