import torch
import torchvision
from PIL import Image
from bable.utils.transforms_utils import ToTensor


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, split, category_id, transforms_config):
        super(BaseDataset, self).__init__()
        assert split in self._get_splits(), 'unknown split %s' % split
        assert category_id in range(len(self._get_categories())), \
            'invalid category_id %d ' % category_id

        self._category_name = self._get_categories()[category_id]
        self._list, self._labels = self._get_list_and_labels(split)
        self._trans_config = transforms_config
        self._transforms = self._get_transforms()

    @property
    def category_name(self):
        return self._category_name

    def _get_splits(self):
        raise NotImplementedError

    def _get_categories(self):
        raise NotImplementedError

    def _get_list_and_labels(self, split):
        raise NotImplementedError

    def _get_transforms(self):
        transforms_list = []

        if self._trans_config.get('color_jitter'):
            transforms_list.append(torchvision.transforms.ColorJitter(
                brightness=self._trans_config.get('brightness'),
                contrast=self._trans_config.get('contrast'),
                saturation=self._trans_config.get('saturation'),
                hue=self._trans_config.get('hue'),
            ))

        if self._trans_config.get('random_horizontal_flip'):
            transforms_list.append(
                torchvision.transforms.RandomHorizontalFlip()
            )

        if self._trans_config.get('resize_size'):
            transforms_list.append(
                torchvision.transforms.Resize(
                    self._trans_config.get('resize_size')
                )
            )

        if self._trans_config.get('random_resized_crop_size'):
            transforms_list.append(
                torchvision.transforms.RandomResizedCrop(
                    self._trans_config.get('random_resized_crop_size'),
                    self._trans_config.get('random_resized_crop_scale'),
                    self._trans_config.get('random_resized_crop_ratio'),
                )
            )

        transforms_list.append(
            ToTensor(is_rgb=self._trans_config.get('is_rgb'))
        )

        if self._trans_config.get('normalize'):
            transforms_list.append(self._trans_config['normalize'])

        return torchvision.transforms.Compose(transforms_list)

    def __getitem__(self, index):
        label = self._labels[index]
        img_p1, img_p2 = self._list[index]
        img1 = Image.open(img_p1)
        img2 = Image.open(img_p2)
        img1 = self._transforms(img1)
        img2 = self._transforms(img2)
        return (img1, img2), label

    def __len__(self):
        return len(self._labels)
