import os
import numpy as np
import scipy.io as sio
from bable.datasets.base_dataset import BaseDataset
from bable.utils.transforms_utils import get_default_transforms_config


class ZapposV1(BaseDataset):
    def __init__(self,
                 split,
                 category_id,
                 trans_config=None,
                 include_equal=False,
                 dataset_dir="/hdd02/zhangyiyang/data/zap50k",
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
            # original: left -> 1, right -> 2, equal -> 3
            # target: left -> -1, right -> 1, equal -> 0
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
        image_names_raw = sio.loadmat(
            opj(annotation_dir, self._image_names_file_name)
        )['imagepath'].flatten()
        image_names = [opj(image_dir, n[0]) for n in image_names_raw]

        image_names = []

        # # 下载下来的数据集路径存在问题，需要修正……
        # # you see this crazy for loop? yes I hate it too.
        # for p in image_names_raw:
        #     this_thing = str(p[0])
        #     this_thing_parts = this_thing.rsplit('/', 1)
        #     if this_thing_parts[0].endswith('.'):
        #         this_thing_parts[0] = this_thing_parts[0][:-1]
        #         this_thing = '/'.join(this_thing_parts)
        #     if "Levi's " in this_thing_parts[0]:
        #         this_thing_parts[0] = this_thing_parts[0].replace(
        #             "Levi's ", "Levi's&#174; ")
        #         this_thing = '/'.join(this_thing_parts)
        #     image_names.append(opj(image_dir, this_thing))

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
        pair_list = [(image_names[ndarray[idx, 0] - 1],
                      image_names[ndarray[idx, 1] - 1])
                     for idx in range(ndarray.shape[0])]

        # labels
        labels = ndarray[:, 3]
        labels = [_convert_winnter(l) for l in labels]

        return pair_list, labels


class ZapposV2(BaseDataset):
    def __init__(self,
                 split,
                 category_id,
                 trans_config=None,
                 include_equal=False,
                 dataset_dir="/hdd02/zhangyiyang/data/zap50k",
                 image_dir_name="ut-zap50k-images-square",
                 annoatation_dir_name="ut-zap50k-data",
                 image_names_file_name="image-path.mat",
                 labels_file_name="zappos-labels.mat",
                 fg_labels_file_name="zappos-labels-fg.mat",
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
        self._labels_file_name = labels_file_name
        self._fg_labels_file_name = fg_labels_file_name

        super(ZapposV2, self).__init__(split, category_id, trans_config)

    def _get_splits(self):
        return ('train', 'test')

    def _get_categories(self):
        return ('open', 'pointy', 'sporty', 'comfort')

    def _get_list_and_labels(self, split):

        def _convert_winnter(old):
            # original: left -> 1, right -> 2, equal -> 3
            # target: left -> -1, right -> 1, equal -> 0
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
        if split == 'train':
            ndarray = sio.loadmat(
                opj(annotation_dir, self._labels_file_name)
            )['mturkOrder'].flatten()[self._category_id]
            if self._include_equal:
                ndarray2 = sio.loadmat(
                    os.path.join(annotation_dir, self._labels_file_name)
                )['mturkEqual'].flatten()[self._category_id]
                ndarray = np.concatenate([ndarray2, ndarray])
        else:
            ndarray = sio.loadmat(
                os.path.join(annotation_dir, self._fg_labels_file_name)
            )['mturkHard'].flatten()[self._category_id]

        # list
        ndarray = ndarray.astype(np.int32)
        pair_list = [(image_names[ndarray[idx, 0] - 1],
                      image_names[ndarray[idx, 1] - 1])
                     for idx in range(ndarray.shape[0])]

        # labels
        labels = ndarray[:, 3]
        labels = [_convert_winnter(l) for l in labels]

        return pair_list, labels
