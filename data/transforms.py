import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size=224):
    """
    Strong augmentations for training.
    Compensates for lack of metadata by forcing visual invariance.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=45, p=0.5, border_mode=0),
        A.HueSaturationValue(
            hue_shift_limit=30,
            sat_shift_limit=40,
            val_shift_limit=30,
            p=0.8
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.4,
            contrast_limit=0.4,
            p=0.8
        ),
        A.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
            p=0.5
        ),
        A.CoarseDropout(
            num_holes=12,
            hole_height=48,
            hole_width=48,
            p=0.5
        ),
        A.GaussNoise(p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])

def get_val_transforms(img_size=224):
    """
    Minimal augmentations for validation/test.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])

