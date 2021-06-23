nominal_batch_size = 64

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    nesterov=True,
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.))

optimizer_config = dict(
    type='Fp16GradAccumulateOptimizerHook',
    nominal_batch_size=nominal_batch_size,
    grad_clip=dict(max_norm=35, norm_type=2),
    loss_scale=dict(
        init_scale=2. ** 16,
        growth_interval=1000))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr_ratio=0.2,
)

load_from = None
resume_from = None

custom_hooks = [
    dict(
        type='DetailedLinearWarmUpHook',
        warmup_iters=10000,
        lr_weight_warmup_ratio=0.,
        lr_bias_warmup_ratio=10.,
        momentum_warmup_ratio=0.95,
        priority='NORMAL'),
    dict(
        type='StateEMAHook',
        momentum=0.9999,
        nominal_batch_size=nominal_batch_size,
        warm_up=10000,
        resume_from=resume_from,
        priority='HIGH')
]

runner = dict(type='EpochBasedRunner', max_epochs=300)

evaluation = dict(interval=1, metric='cowa')

checkpoint_config = dict(interval=5)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]

cudnn_benchmark = True
