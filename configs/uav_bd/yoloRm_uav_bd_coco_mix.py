_base_ = [
    '../_base_/models/yoloRm.py',
    '../_base_/datasets/uav_bd_coco_mix.py',
    '../_base_/schedules/yoloR_schedule.py'
]
data = dict(
    samples_per_gpu=20,
    workers_per_gpu=4)

model = dict(train_cfg=dict(num_obj_per_image=2.2))
