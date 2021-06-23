_base_ = [
    '../_base_/models/yoloRm.py',
    '../_base_/datasets/uav_bd.py',
    '../_base_/schedules/yoloR_schedule.py'
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=0)
checkpoint_config = dict(interval=1)
