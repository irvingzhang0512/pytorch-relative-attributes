import numpy as np


class ScoresAccuracyTool:
    def __init__(self):
        self._positive = 0
        self._total = 0

    def update(self, s1, s2, labels):
        self._total += len(labels)
        s = s1 - s2
        self._positive += \
            len(np.intersect1d(np.where(s > 0), np.where(labels == -1))) + \
            len(np.intersect1d(np.where(s < 0), np.where(labels == 1)))

    def accuracy(self):
        return self._positive * 1.0 / self._total

    def reset(self):
        self._positive = 0
        self._total = 0


class MeanTool:
    def __init__(self):
        self._value = 0.
        self._cnt = 0.

    def update(self, value, cnt=1.):
        self._value += value
        self._cnt += cnt

    def mean(self):
        return .0 if self._cnt == 0. else self._value / self._cnt

    def reset(self):
        self._value = 0.
        self._cnt = 0.
