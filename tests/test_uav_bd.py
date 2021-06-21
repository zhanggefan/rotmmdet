from mmdetrot.datasets import UAVBD
import matplotlib.pyplot as plt

img_norm_cfg = dict(mean=[114, 114, 114], std=[255, 255, 255], to_rgb=True)
train_pipeline = [
    dict(
        type='MosaicPipeline',
        individual_pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='ResizeR', img_scale=(342, 342), keep_ratio=True),
            dict(type='RandomFlipR', flip_ratio=0.0),
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


def test_uav_bd():
    dataset = UAVBD(
        ann_file='train.txt',
        data_root='data/uav-bd',
        img_prefix='train',
        pipeline=train_pipeline,
    )
    for i in dataset:
        plt.imshow(i['img'].data.permute(1, 2, 0).numpy())
        plt.show()
    pass


if __name__ == '__main__':
    test_uav_bd()
