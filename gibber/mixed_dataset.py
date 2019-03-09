# For feeding Torch batches of tensors and lists of strings, which is needed with ELMo.

import torch

class MixedDataset():

    def __init__(self, batch_size, *sources):
        assert len(sources) > 0

        if isinstance(sources[0], torch.Tensor):
            first_len = sources[0].size(0)
        elif isinstance(sources[0], list):
            first_len = len(sources[0])

        for src in sources:
            if isinstance(src, torch.Tensor):
                assert src.size(0) == first_len
            elif isinstance(src, list):
                assert len(src) == first_len
            else:
                raise NotImplementedError('{} is not of an implemented (list of tensor)'.format(src))

        self.sources = sources
        self.srcs_len = first_len

        self.batch_size = batch_size
        self.batch_n = 0

    def __iter__(self):
        return self

    def __getitem__(self, index):
        print(index)
        return tuple(src[index] for src in self.sources)

    def __len__(self):
        return self.srcs_len

    def __next__(self):
        if self.batch_n >= self.srcs_len / self.batch_size:
            raise StopIteration
        src_batches = []
        for src in self.sources:
            src_batches.append(src[self.batch_n*self.batch_size:(self.batch_n+1)*self.batch_size])
        self.batch_n += 1
        return src_batches
