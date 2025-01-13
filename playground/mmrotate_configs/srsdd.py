# dataset settings
dataset_type = 'mmdet.CocoDataset'
data_root = 'playground/data/SRSDD/'

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(
        type='mmdet.LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ConvertMask2BoxType', box_type='rbox'),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    # avoid bboxes being resized
    dict(
        type='mmdet.LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ConvertMask2BoxType', box_type='qbox'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'instances'))
]

metainfo = dict(
    classes=('Cell-Container', 'Container', 'Dredger', 'Fishing', 'LawEnforce', 'ore-oil'))

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img='test/images/'),
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = val_dataloader

# val_evaluator = dict(type='DOTAMetric', metric='mAP')
val_evaluator = dict(type='RotatedCocoMetric', metric='bbox', classwise=True)
test_evaluator = val_evaluator