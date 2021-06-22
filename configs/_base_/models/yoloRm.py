model = dict(
    type='SingleStageDetectorR',
    backbone=dict(
        type='DarknetCSP',
        scale='v4m5p',
        out_indices=[3, 4, 5],
        act_cfg=dict(type='Swish'),
        csp_act_cfg=dict(type='Swish'),
    ),
    neck=dict(
        type='YOLOV4Neck',
        in_channels=[192, 384, 384],
        out_channels=[192, 384, 768],
        csp_repetition=1,
        act_cfg=dict(type='Swish'),
        csp_act_cfg=dict(type='Swish'),
    ),
    bbox_head=dict(
        type='YOLORHead',
        class_agnostic=True,
        num_classes=1,
        in_channels=[192, 384, 768],
        act_cfg=dict(type='Swish'),
        loss_bbox=dict(type='GDLoss', loss_type='gwd',
                       loss_weight=3.2)
    ),
    train_cfg=dict(num_obj_per_image=2.8),
    test_cfg=dict(
        min_bbox_size=0,
        nms_pre=-1,
        score_thr=0.001,
        nms=dict(type='nms_rotated', iou_threshold=0.3),
        max_per_img=300),
)
