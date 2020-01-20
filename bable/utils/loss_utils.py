import torch
import torch.nn.functional as F

"""
original labels
left -> -1, right -> 1, equal -> 0

scores, labels, weights
scores: including s1 and s2, both shape [batch_size]
labels: shape like s1 & s2
weights: float
"""


def ranknet(s1, s2, labels, epsilon=1e-7):
    """ranknet loss"""
    s1.squeeze_()
    s2.squeeze_()
    labels = labels.float()
    labels[labels == 0] = 0.5
    labels[labels == -1] = 0
    s = torch.sigmoid(s2 - s1).clamp(epsilon, 1-epsilon)
    # print(s, labels)
    return F.binary_cross_entropy(s, labels)


def dra(s1, s2, labels):
    s1.squeeze_()
    s2.squeeze_()
    s = s1 - s2
    dra_loss = torch.empty(labels.shape).to("cuda:0")
    dra_loss[labels == 0] = torch.pow(s[labels == 0], 2) * 0.5
    dra_loss[labels != 0] = 1.0 + labels[labels != 0].float() * s
    dra_loss[dra_loss < 0] = 0

    return dra_loss.mean()
