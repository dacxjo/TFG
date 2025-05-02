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

