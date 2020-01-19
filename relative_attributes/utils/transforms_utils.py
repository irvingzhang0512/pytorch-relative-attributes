import torch
import torchvision
from torchvision.transforms import functional as F


def get_default_transforms_config(training=True):
    return {
        # ToTensor
        "is_rgb": True,

        # ColorJitter
        "brightness": .0,
        "contrast": .0,
        "saturation": .0,
        "hue": .0,
        
        # RandomHorizontalFlip
        "random_horizontal_flip": training,

        # RandomResizedCrop
        "random_resized_crop_size": (224, 224) if training else None,
        "random_resized_crop_scale": (0.75, 1.),
        "random_resized_crop_ratio": (3.0 / 4, 4.0 / 3),
        
        # Resize
        "resize_size": None if training else (256, 256),
    }


class ToTensor(object):
    def __init__(self, is_rgb=True):
        self._is_rgb = is_rgb

    def __call__(self, pic):
        if not self._is_rgb:
            pic = pic[:, :, ::-1]
        return F.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'
