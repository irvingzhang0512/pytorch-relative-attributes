from bable.datasets.zappos_dataset import ZapposV1, ZapposV2
from bable.datasets.zappos_dataset import ZapposPredictDataset
from bable.datasets.lfw10_dataset import LFW10Dataset


def build_dataset(dataset_type, **kwargs):
    if dataset_type == 'zappos_v1':
        # params: split, category_id, trans_config, include_equal
        return ZapposV1(**kwargs)
    elif dataset_type == 'zappos_v2':
        # params: split, category_id, trans_config, include_equal
        return ZapposV2(**kwargs)
    elif dataset_type == 'zappos_predict':
        # params: min_height, min_witdh, is_bgr
        return ZapposPredictDataset(**kwargs)
    elif dataset_type == 'lfw10':
        # params: split, category_id, trans_config, include_equal
        return LFW10Dataset(**kwargs)
    raise ValueError('unknown dataset type %s' % dataset_type)
