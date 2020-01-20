from bable.datasets.zappos_dataset import ZapposV1, ZapposV2


def build_dataset(dataset_type, **kwargs):
    if dataset_type == 'zappos_v1':
        return ZapposV1(**kwargs)
    elif dataset_type == 'zappos_v2':
        return ZapposV2(**kwargs)
    raise ValueError('unknown dataset type %s' % dataset_type)
