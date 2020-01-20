from torchvision import transforms
from torchvision.transforms import functional as F


def get_default_transforms_config(training=True):
    return {
        # ToTensor
        "is_rgb": True,

        # ColorJitter
        "color_jitter": training,
        "brightness": .1,
        "contrast": .1,
        "saturation": .1,
        "hue": .1,

        # RandomHorizontalFlip
        "random_horizontal_flip": training,

        # RandomResizedCrop
        "random_resized_crop_size": (224, 224) if training else None,
        "random_resized_crop_scale": (0.75, 1.),
        "random_resized_crop_ratio": (3.0 / 4, 4.0 / 3),

        # Resize
        "resize_size": None if training else(256, 256),

        # Nomalize
        "normalize": transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
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
