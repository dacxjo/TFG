import albumentations as A


def get_transforms(augment=False, target_size=224):
    if augment:
        return A.Compose([
            A.SmallestMaxSize(max_size=target_size, p=1.0),
            A.RandomResizedCrop(size=(target_size, target_size), scale=(0.85, 1.0), ratio=(0.9, 1.1), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.SmallestMaxSize(max_size=target_size, p=1.0),
            A.CenterCrop(height=target_size, width=target_size, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.ToTensorV2(),
        ])


def get_transforms_aggressive(augment=False, target_size=224):
    if augment:
        return A.Compose([
            A.SmallestMaxSize(max_size=target_size, p=1.0),
            A.RandomResizedCrop(
                size=(target_size, target_size),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=1.0
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1),
                    contrast_limit=(-0.1, 0.1),
                    p=0.5
                ),
            ], p=0.8),
            A.GaussNoise(std_range=(0.02, 0.1),
                         mean_range=(-0.01, 0.01),
                         per_channel=False,
                         noise_scale_factor=0.5,
                         p=0.3),
            A.GridDropout(
                ratio=0.1,
                unit_size_range=(16, 32),
                random_offset=True,
                fill='inpaint_telea',
                p=0.2
            ),
            A.ElasticTransform(
                alpha=1,
                sigma=30,
                p=0.3
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.SmallestMaxSize(max_size=target_size, p=1.0),
            A.CenterCrop(height=target_size, width=target_size, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.ToTensorV2(),
        ])
