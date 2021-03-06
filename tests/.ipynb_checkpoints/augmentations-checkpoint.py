from typing import Dict

import albumentations as albu
import numpy as np
import torch
import cv2


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor_copy = tensor.clone()
        for t, m, s in zip(tensor_copy, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor_copy


output_format = {
    "none": lambda array: array,
    "float": lambda array: torch.FloatTensor(array),
    "long": lambda array: torch.LongTensor(array),
}

normalization = {
    "none": lambda array: array,
    "default": lambda array: albu.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )(image=array)["image"],
    "div255": lambda array: array / 255,
    "binary": lambda array: np.array(array > 0, np.float32)
}

denormalization = {
    "none": lambda array: array,
    "default": lambda array: UnNormalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(array),
    "div255": lambda array: array * 255,
}

augmentations = {
    "strong": albu.Compose(
        [
            albu.HorizontalFlip(),
            albu.ShiftScaleRotate(
                shift_limit=0.3, scale_limit=0.2, rotate_limit=90, p=0.4
            ),
            albu.GaussNoise(),
            albu.OneOf(
                [
                    albu.CLAHE(clip_limit=2),
                    albu.IAASharpen(),
                    albu.IAAEmboss(),
                    albu.RandomBrightnessContrast(),
                    albu.RandomGamma(),
                    albu.MedianBlur(),
                ],
                p=0.5,
            ),
        ]
    ),
    "weak": albu.Compose([albu.HorizontalFlip()]),
    "none": albu.Compose([]),
}

size_augmentations = {
    "none": lambda size: albu.NoOp(),
    "resize": lambda size: albu.Resize(height=size, width=size,  interpolation=cv2.INTER_CUBIC),
    "random": lambda size: albu.RandomCrop(size, size),
    "center": lambda size: albu.CenterCrop(size, size),
    "random_crop_and_resize": lambda size: albu.OneOf([
        albu.RandomCrop(size, size),
        albu.Resize(height=size, width=size, interpolation=cv2.INTER_CUBIC)
    ], p=1),
    "crop": lambda size: albu.RandomCrop(size, size)
}


def get_transforms(config: Dict):
    size = config["size"]
    scope = config.get("augmentation_scope", "none")
    size_transform = config.get("size_transform", "none")

    images_normalization = config.get("images_normalization", "default")
    masks_normalization = config.get("masks_normalization", "binary")

    images_output_format_type = config.get("images_output_format_type", "float")
    masks_output_format_type = config.get("masks_output_format_type", "float")

    pipeline = albu.Compose(
        [
            # albu.PadIfNeeded(p=1, min_height=size, min_width=size),
            augmentations[scope],
            size_augmentations[size_transform](size),
        ]
    )

    def process(image, mask):
        r = pipeline(image=image, mask=mask)

        transformed_image = output_format[images_output_format_type](
            normalization[images_normalization](r["image"])
        )

        transformed_mask = output_format[masks_output_format_type](
            normalization[masks_normalization](r["mask"])
        )

        return transformed_image, transformed_mask

    return process
