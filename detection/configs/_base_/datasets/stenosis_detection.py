# dataset settings

dataset_type = "CocoStenosisBinaryDataset"
fold = "FOLD0"
data_root = "/ai/mnt/data/stenosis/selected_small/Binary/FOLD0/"
dataset_name = "STENOSIS_BINARY"
train_ann_file = "annotations/train_binary.json"
val_ann_file = "annotations/val_binary.json"
train_data_prefix = dict(img="train/")
val_data_prefix = dict(img="val/")


img_norm_cfg = dict(
    mean=[144.5754766729963, 144.5754766729963, 144.5754766729963],
    std=[55.8710224233549, 55.8710224233549, 55.8710224233549],
    to_rgb=True,
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", img_scale=(1333, 800), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/train_binary.json",
        img_prefix=data_root + "train/",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/val_binary.json",
        img_prefix=data_root + "val/",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/val_binary.json",
        img_prefix=data_root + "val/",
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=1, metric="bbox", classwise=True)
