_base_ = [
    '../_base_/models/yoloRm.py',
    '../_base_/datasets/uav_bd.py',
    '../_base_/schedules/yoloR_schedule.py'
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4)

model = dict(
    backbone=dict(
        norm_cfg=dict(
            type='SyncBN', requires_grad=True, eps=0.001, momentum=0.03)
    ),
    neck=dict(
        norm_cfg=dict(
            type='SyncBN', requires_grad=True, eps=0.001, momentum=0.03)
    )
)

optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
fp16 = dict(loss_scale='dynamic')
