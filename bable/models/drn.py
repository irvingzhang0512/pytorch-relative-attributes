from torch import nn
from torchvision import models
from bable.models.base_siamese_model import BaseSiameseModel


class DRN(BaseSiameseModel):
    def __init__(self, extractor_type):
        if extractor_type == 'vgg16':
            extractor = models.vgg16(pretrained=True)
            extractor.classifier._modules["6"] = nn.Linear(4096, 1)
        elif extractor_type == 'inception_v3':
            extractor = models.inception_v3(pretrained=True,
                                            aux_logits=False)
            extractor.fc = nn.Linear(2048, 1)
        elif extractor_type == 'googlenet':
            extractor = models.googlenet(pretrained=True)
            extractor.fc = nn.Linear(1024, 1)
        else:
            raise ValueError('unknown extractor type %s' % extractor_type)

        super(DRN, self).__init__(extractor)
