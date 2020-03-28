import os
import platform
import pandas as pd
from bable.datasets.base_dataset import BasePredictDataset
from bable.datasets.base_dataset import BaseSiameseDataset
from bable.utils.transforms_utils import get_default_transforms_config


if 'Windows' in platform.platform():
    BASE_DATASET = "F:\\data\\BaiduStreetView"
else:
    BASE_DATASET = "/hdd02/zhangyiyang/data/BaiduStreetView"


CATEGORIES = (
    'beautiful', 'safety'
)

SPLITS = ('train', 'val')


class BaiduStreetViewDataset(BaseSiameseDataset):
    def __init__(self,
                 split,
                 category_id,
                 trans_config=None,
                 include_equal=False,
                 dataset_dir=BASE_DATASET,
                 image_dir_name='images',
                 annoatation_file_name='annos.csv',
                 ):
        if trans_config is None:
            trans_config = get_default_transforms_config()
        self._split = split
        self._category_id = category_id
        self._include_equal = include_equal
        self._dataset_dir = dataset_dir
        self._image_dir_name = image_dir_name
        self._annoatation_file_name = annoatation_file_name
        super(BaiduStreetViewDataset, self).__init__(
            split, category_id, trans_config)

    def _get_splits(self):
        return SPLITS

    def _get_categories(self):
        return CATEGORIES

    def _get_list_and_labels(self, split):

        def _get_winner_label(cur_winner):
            if cur_winner == 'right':
                return 1
            elif cur_winner == 'left':
                return -1
            return 0

        df = pd.read_csv(os.path.join(
            self._dataset_dir,
            '{}_{}_equal.csv'.format(self.category_name, split)
        ))
        image_dir = os.path.join(self._dataset_dir, self._image_dir_name)
        df = df[df.category == self.category_name]
        if not self._include_equal:
            df = df[df.winner != 'equal']

        winner = df.winner
        labels = [_get_winner_label(cur_winner) for cur_winner in winner]

        left_id = df.left_id
        right_id = df.right_id
        pair_list = [(
            os.path.join(image_dir, left_id[i] + '.jpg'),
            os.path.join(image_dir, right_id[i] + '.jpg')
        ) for i in range(len(labels))]

        return pair_list, labels


class BaiduStreetViewPredictDataset(BasePredictDataset):
    def __init__(self,
                 min_height=224,
                 min_width=224,
                 is_bgr=False,
                 dataset_dir=BASE_DATASET,
                 image_dir_name='images',
                 broken_file_name=None,
                 ):
        self._dataset_dir = dataset_dir
        self._image_dir_name = image_dir_name
        self._broken_file_name = broken_file_name
        super(BaiduStreetViewPredictDataset, self).__init__(
            min_height, min_width, is_bgr
        )

    def _get_image_full_paths(self):
        image_dir = os.path.join(self._dataset_dir, self._image_dir_name)
        file_names = os.listdir(image_dir)
        broken_files = []
        if self._broken_file_name is not None:
            with open(os.path.join(
                    self._dataset_dir,
                    self._broken_file_name), 'r') as f:
                broken_files = f.readlines()
            broken_files = [name.replace('\n', '') for name in broken_files]
        image_list = [os.path.join(image_dir, fname)
                      for fname in file_names if fname not in broken_files]
        return image_list
