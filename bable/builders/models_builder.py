from bable.models.drn import DRN


def build_model(model_type, **kwargs):
    if model_type == 'drn':
        return DRN(kwargs['extractor_type'])
    raise ValueError('unknown model type %s' % model_type)
