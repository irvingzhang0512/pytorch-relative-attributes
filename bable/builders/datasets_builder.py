from bable.datasets.zappos_dataset import ZapposV1, ZapposV2
from bable.datasets.zappos_dataset import ZapposPredictDataset
from bable.datasets.lfw10_dataset import LFW10Dataset, LFW10PredictDataset
from bable.datasets.pubfig_dataset import PubfigDataset, PubfigPredictDataset
from bable.datasets.osr_dataset import OSRDataset, OSRPredictDataset
from bable.datasets.place_pulse_dataset import PlacePulseDataset
from bable.datasets.place_pulse_dataset import PlacePulsePredictDataset


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
    elif dataset_type == 'lfw10_predict':
        # params: min_height, min_witdh, is_bgr
        return LFW10PredictDataset(**kwargs)
    elif dataset_type == 'pubfig':
        # params: split, category_id, trans_config, include_equal
        return PubfigDataset(**kwargs)
    elif dataset_type == 'pubfig_predict':
        # params: min_height, min_witdh, is_bgr
        return PubfigPredictDataset(**kwargs)
    elif dataset_type == 'osr':
        # params: split, category_id, trans_config, include_equal
        return OSRDataset(**kwargs)
    elif dataset_type == 'osr_predict':
        # params: min_height, min_witdh, is_bgr
        return OSRPredictDataset(**kwargs)
    elif dataset_type == 'place_pulse':
        # params: split, category_id, trans_config, include_equal
        return PlacePulseDataset(**kwargs)
    elif dataset_type == 'place_pulse_predict':
        # params: min_height, min_witdh, is_bgr
        return PlacePulsePredictDataset(**kwargs)
    raise ValueError('unknown dataset type %s' % dataset_type)
