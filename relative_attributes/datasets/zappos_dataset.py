import os
import numpy as np
import scipy.io as sio
from relative_attributes.datasets.base_datasets import BaseDataset
from relative_attributes.utils.transforms_utils import \
    get_default_transforms_config


class ZapposV1(BaseDataset):
    def __init__(self,
                 split,
                 category_id,
                 trans_config=None,
                 include_equal=False,
                 dataset_dir="F:\\data\\zap50k",
                 image_dir_name="ut-zap50k-images-square",
                 annoatation_dir_name="ut-zap50k-data",
                 image_names_file_name="image-path.mat",
                 pairs_file_name="train-test-splits-pairs.mat",
                 ):
        if trans_config is None:
            trans_config = get_default_transforms_config()
        self._split = split
        self._category_id = category_id
        self._include_equal = include_equal
        self._dataset_dir = dataset_dir
        self._image_dir_name = image_dir_name
        self._annoatation_dir_name = annoatation_dir_name
        self._image_names_file_name = image_names_file_name
        self._pairs_file_name = pairs_file_name

        super(ZapposV1, self).__init__(split, category_id, trans_config)

    def _get_splits(self):
        return ('train', 'test')

    def _get_categories(self):
        return ('open', 'pointy', 'sporty', 'comfort')

    def _get_list_and_labels(self, split):

        def _convert_winnter(old):
            # lfw original: left -> 1, right -> 2, equal -> 3
            # this project: left -> -1, right -> 1, equal -> 0
            if old == 3:
                return 0
            if old == 1:
                return -1
            return 1

        # dirs
        opj = os.path.join
        image_dir = opj(self._dataset_dir, self._image_dir_name)
        annotation_dir = opj(
            self._dataset_dir, self._annoatation_dir_name)

        # image names list
        image_names_file = sio.loadmat(
            opj(annotation_dir, self._image_names_file_name)
        )
        image_names = image_names_file['imagepath'].flatten()
        image_names = [opj(image_dir, n[0]) for n in image_names]

        # pairs and ndarray
        pairs_file = sio.loadmat(opj(annotation_dir, self._pairs_file_name))
        if self._split == 'train':
            ndarray = pairs_file['trainPairsAll'].flatten()[
                self._category_id].flatten()[0]
        else:
            ndarray = pairs_file['testPairsAll'].flatten()[
                self._category_id].flatten()[0]
        ndarray = ndarray.astype(np.int32)
        if not self._include_equal:
            ids = np.where(ndarray[:, 3] != 3)[0]
            ndarray = ndarray[ids]

        # list
        pair_list = [(image_names[ndarray[idx, 0]],
                      image_names[ndarray[idx, 1]])
                     for idx in range(ndarray.shape[0])]

        # labels
        labels = ndarray[:, 3]
        labels = [_convert_winnter(l) for l in labels]

        return pair_list, labels
