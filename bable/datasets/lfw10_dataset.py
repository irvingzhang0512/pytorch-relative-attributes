import os
import platform
import numpy as np
import scipy.io as sio
from bable.datasets.base_dataset import BasePredictDataset
from bable.datasets.base_dataset import BaseSiameseDataset
from bable.utils.transforms_utils import get_default_transforms_config


if 'Windows' in platform.platform():
    BASE_DATASET = "F:\\data\\LFW10"
else:
    BASE_DATASET = "/hdd02/zhangyiyang/data/LFW10"

NUM_SAMPLES = 500

CATEGORIES = (
    'baldhead', 'darkhair', 'eyesopen', 'goodlooking',
    'masculinelooking', 'mouthopen', 'smile', 'vforehead',
    'v_teeth', 'young'
)

SPLITS = ('train', 'test')


class LFW10Dataset(BaseSiameseDataset):
    def __init__(self,
                 split,
                 category_id,
                 trans_config=None,
                 include_equal=False,
                 dataset_dir=BASE_DATASET,
                 image_dir_name='images',
                 annoatation_dir_name='annotations',
                 ):
        if trans_config is None:
            trans_config = get_default_transforms_config()
        self._split = split
        self._category_id = category_id
        self._include_equal = include_equal
        self._dataset_dir = dataset_dir
        self._image_dir_name = image_dir_name
        self._annoatation_dir_name = annoatation_dir_name
        super(LFW10Dataset, self).__init__(
            split, category_id, trans_config)

    def _get_splits(self):
        return SPLITS

    def _get_categories(self):
        return CATEGORIES

    def _get_list_and_labels(self, split):
        def _convert_winnter(old):
            # original: left -> 0, right -> 1, equal -> 2
            # target: left -> -1, right -> 1, equal -> 0
            if old == 0:
                return - 1
            if old == 2:
                return 0
            return old

        anno_file_path = os.path.join(
            self._dataset_dir, self._annoatation_dir_name,
            "{}{}.mat".format(self.category_name, split)
        )
        mat_dataset = sio.loadmat(anno_file_path)
        img_names = mat_dataset['images_compare']
        attr_values = mat_dataset['attribute_strengths']
        left_ids = []
        right_ids = []
        winners = np.argmax(attr_values[:, 1:], axis=1)
        winners = np.array([_convert_winnter(w) for w in winners])
        for i in range(NUM_SAMPLES):
            cur_image_names = img_names[i]
            left_ids.append(str(cur_image_names[1][0]))
            right_ids.append(str(cur_image_names[2][0]))

        image_dir = os.path.join(self._dataset_dir, self._image_dir_name)
        left_paths = [os.path.join(image_dir, l) for l in left_ids]
        right_paths = [os.path.join(image_dir, r) for r in right_ids]
        left_paths = np.array(left_paths)
        right_paths = np.array(right_paths)
        if not self._include_equal:
            ids = np.where(winners != 0)
            winners = winners[ids]
            left_paths = left_paths[ids]
            right_paths = right_paths[ids]
        pair_list = [(left_paths[i], right_paths[i])
                     for i in range(len(winners))]
        return pair_list, winners


class LFW10PredictDataset(BasePredictDataset):
    def __init__(self,
                 min_height=224,
                 min_width=224,
                 is_bgr=False,
                 dataset_dir=BASE_DATASET,
                 image_dir_name='images',
                 ):
        self._dataset_dir = dataset_dir
        self._image_dir_name = image_dir_name
        super(LFW10PredictDataset, self).__init__(
            min_height, min_width, is_bgr
        )

    def _get_image_full_paths(self):
        image_dir = os.path.join(self._dataset_dir, self._image_dir_name)
        file_names = os.listdir(image_dir)
        image_list = [os.path.join(image_dir, fname) for fname in file_names]
        image_list = [i for i in image_list if os.path.exists(i)]
        return image_list
