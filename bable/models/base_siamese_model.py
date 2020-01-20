from torch import nn


class BaseSiameseModel(nn.Module):
    def __init__(self, ranker):
        super(BaseSiameseModel, self).__init__()
        self.ranker = ranker

    def forward(self, img1, img2):
        out1 = self.ranker(img1)
        out2 = self.ranker(img2)
        return out1, out2
