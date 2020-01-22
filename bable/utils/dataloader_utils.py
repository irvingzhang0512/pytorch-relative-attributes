import torch
from torch.utils.data import DataLoader


class PrefetchDataLoader(DataLoader):
    def __iter__(self):
        from prefetch_generator import BackgroundGenerator
        return BackgroundGenerator(super().__iter__())


class DataPrefetcher():
    def __init__(self, loader):
        self.dataset = loader.dataset
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            (self.next_input1, self.next_input2), self.next_target = next(self.loader)
        except StopIteration:
            self.next_input1 = None
            self.next_input2 = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input1 = self.next_input1.cuda(non_blocking=True)
            self.next_input2 = self.next_input2.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = (self.next_input1, self.next_input2), self.next_target
        self.preload()
        return batch

    def __next__(self):
        return self.next()
