from bable.utils import loss_utils


def build_loss(loss_type):
    if loss_type == 'ranknet':
        return loss_utils.ranknet
    elif loss_type == 'dra':
        return loss_utils.dra
    raise ValueError('unknown loss type %s' % loss_type)
