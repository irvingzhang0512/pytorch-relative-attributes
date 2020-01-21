import os
import itertools
import platform
import numpy as np
import scipy.io as sio
from bable.datasets.base_dataset import BasePredictDataset
from bable.datasets.base_dataset import BaseSiameseDataset
from bable.utils.transforms_utils import get_default_transforms_config


if 'Windows' in platform.platform():
    BASE_DATASET = "F:\\data\\OSR"
else:
    BASE_DATASET = "/hdd02/zhangyiyang/data/OSR"

TEST_FRACTION = 0.05

CATEGORIES = (
    'natural', 'open', 'perspective', 'size-large',
    'diagonal-plane', 'depth-close'
)

SPLITS = ('train', 'test')


class OSRDataset(BaseSiameseDataset):
    def __init__(self,
                 split,
                 category_id,
                 trans_config=None,
                 include_equal=False,
                 dataset_dir=BASE_DATASET,
                 image_dir_name='spatial_envelope_256x256_static_8outdoorcategories',
                 annoatation_file_name='data.mat',
                 ):
        if trans_config is None:
            trans_config = get_default_transforms_config()
        self._split = split
        self._category_id = category_id
        self._include_equal = include_equal
        self._dataset_dir = dataset_dir
        self._image_dir_name = image_dir_name
        self._annoatation_file_name = annoatation_file_name
        super(OSRDataset, self).__init__(
            split, category_id, trans_config)

    def _get_splits(self):
        return SPLITS

    def _get_categories(self):
        return CATEGORIES

    def _get_list_and_labels(self, split):
        images_path = os.path.join(self._dataset_dir, self._image_dir_name)
        data_file = sio.loadmat(
            os.path.join(self._dataset_dir, self._annoatation_file_name),
            appendmat=False
        )

        im_names = data_file['im_names'].squeeze()
        image_list = [os.path.join(
            images_path, im_names[i][0]) for i in range(len(im_names))]
        class_labels = data_file['class_labels'][:, 0]
        used_for_training = data_file['used_for_training'][:, 0]
        if split == 'train':
            condition = used_for_training
        else:
            condition = used_for_training - 1

        X = np.arange(len(im_names), dtype=np.int)
        y = np.zeros(
            (len(im_names), len(self._get_categories())), dtype=np.int)
        for i in range(len(im_names)):
            y[i, :] = data_file['relative_ordering'][:, class_labels[i] - 1]
        XX = X[np.where(condition)]
        yy = y[np.where(condition)]

        idxs = list(itertools.combinations(range(len(XX)), 2))
        pairs = np.zeros((len(idxs), 2), dtype=np.int)
        labels = np.zeros((len(idxs),), dtype=np.float32)
        for cnt, ij in enumerate(idxs):
            i, j = ij
            pairs[cnt][0] = XX[i]
            pairs[cnt][1] = XX[j]
            if yy[i, self._category_id] == yy[j, self._category_id]:
                labels[cnt] = 0
            elif yy[i, self._category_id] > yy[j, self._category_id]:
                labels[cnt] = -1
            else:
                labels[cnt] = 1

        if split != 'train':
            # Since the number of test_pairs are very large,
            # nearly 3 millions, we only sample 5% of them
            # for the actual evaluation
            the_test_length = len(labels)
            fraction_of_the_length = int(the_test_length * TEST_FRACTION)

            indices = np.arange(len(labels), dtype=np.int)
            np.random.shuffle(indices)

            pairs = pairs[indices][:fraction_of_the_length]
            labels = labels[indices][:fraction_of_the_length]

        pair_list = [(image_list[pairs[i][0]], image_list[pairs[i][1]])
                     for i in range(len(labels))]
        return pair_list, labels


class OSRPredictDataset(BasePredictDataset):
    def __init__(self,
                 min_height=224,
                 min_width=224,
                 is_bgr=False,
                 dataset_dir=BASE_DATASET,
                 image_dir_name='spatial_envelope_256x256_static_8outdoorcategories',
                 ):
        self._dataset_dir = dataset_dir
        self._image_dir_name = image_dir_name
        super(OSRPredictDataset, self).__init__(
            min_height, min_width, is_bgr
        )

    def _get_image_full_paths(self):
        image_dir = os.path.join(self._dataset_dir, self._image_dir_name)
        file_names = os.listdir(image_dir)
        image_list = [os.path.join(image_dir, fname) for fname in file_names]
        return image_list
