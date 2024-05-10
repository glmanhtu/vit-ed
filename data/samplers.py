# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import math
from typing import Iterator, Optional
from itertools import chain

import numpy as np
import torch
from pytorch_metric_learning.utils import common_functions
from torch.utils.data import Sampler, DistributedSampler, Dataset
import torch.distributed as dist


class DistributedRepeatSampler(DistributedSampler):

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, repeat=1):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.repeat = repeat

    def __iter__(self):
        all_indicates = []
        for i in range(self.repeat):
            if self.shuffle:
                # deterministically shuffle based on epoch and seed
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
            else:
                indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

            if not self.drop_last:
                # add extra samples to make it evenly divisible
                padding_size = self.total_size - len(indices)
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
            else:
                # remove tail of data to make it evenly divisible.
                indices = indices[:self.total_size]
            assert len(indices) == self.total_size

            # subsample
            indices = indices[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            all_indicates += indices
        return iter(all_indicates)

    def __len__(self) -> int:
        return self.num_samples * self.repeat


class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedIndicatesSampler(Sampler):
    r"""
    DistributedOrderedIndicatesSampler is different from DistributedSampler.
    It does NOT add extra samples to make it evenly divisible.
    DistributedEvalSampler should NOT be used for training. The distributed processes could hang forever.
    See this issue for details: https://github.com/pytorch/pytorch/issues/22584

    DistributedOrderedIndicatesSampler is for evaluation purpose where synchronization does not happen every epoch.
    Synchronization should be done outside the dataloader loop.

    Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        indexes: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
    """

    def __init__(self, indexes, num_replicas, rank):
        super().__init__(None)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        n_samples_per_rep = math.ceil(len(indexes) / self.num_replicas)
        indices = torch.split(indexes, n_samples_per_rep)
        sizes = [0]
        for i in range(1, len(indices)):
            if indices[i][0] == indices[i - 1][-1]:
                sizes.append(indices[i][0].item() - 1)
            else:
                sizes.append(indices[i][0].item())

        sizes.append(indexes[-1].item() + 1)
        item_count_per_rank = []
        for i in range(len(sizes) - 1):
            mask_items = torch.greater_equal(indexes, sizes[i])
            mask_items = torch.logical_and(mask_items, torch.less_equal(indexes, sizes[i + 1]))
            item_count_per_rank.append(torch.sum(mask_items))

        self.max_items_count_per_gpu = max(item_count_per_rank)
        all_indicates = []
        n_items = []
        for i in range(len(sizes) - 1):
            all_indicates.append(torch.arange(sizes[i], sizes[i + 1]))
            n_items.append(torch.sum(indexes < sizes[i + 1]))
        self.samples = all_indicates[self.rank]
        self.n_items = n_items
        self.num_samples = len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return self.num_samples


class DistributedEvalSampler(Sampler):
    r"""
    DistributedEvalSampler is different from DistributedSampler.
    It does NOT add extra samples to make it evenly divisible.
    DistributedEvalSampler should NOT be used for training. The distributed processes could hang forever.
    See this issue for details: https://github.com/pytorch/pytorch/issues/22584
    shuffle is disabled by default

    DistributedEvalSampler is for evaluation purpose where synchronization does not happen every epoch.
    Synchronization should be done outside the dataloader loop.

    Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.

    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0, repeat=1):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.repeat = repeat
        # self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        # self.total_size = self.num_samples * self.num_replicas
        self.total_size = len(self.dataset)         # true value without extra samples
        indices = list(range(self.total_size))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)             # true value without extra samples

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        all_indices = []
        for i in range(self.repeat):
            if self.shuffle:
                # deterministically shuffle based on epoch and seed
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = list(range(len(self.dataset)))


            # # add extra samples to make it evenly divisible
            # indices += indices[:(self.total_size - len(indices))]
            # assert len(indices) == self.total_size

            # subsample
            indices = indices[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            all_indices += indices
        return iter(all_indices)

    def __len__(self):
        return self.num_samples * self.repeat

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): _epoch number.
        """
        self.epoch = epoch


class MPerClassSampler(Sampler):
    """
    At every iteration, this will return m samples per class. For example,
    if dataloader's batchsize is 100, and m = 5, then 20 classes with 5 samples
    each will be returned
    """

    def __init__(self, labels, m, batch_size=None, length_before_new_iter=100000):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.m_per_class = int(m)
        self.batch_size = int(batch_size) if batch_size is not None else batch_size
        self.labels_to_indices = common_functions.get_labels_to_indices(labels)
        self.labels = list(self.labels_to_indices.keys())
        self.length_of_single_pass = self.m_per_class * len(self.labels)
        self.list_size = length_before_new_iter
        if self.batch_size is None:
            if self.length_of_single_pass < self.list_size:
                self.list_size -= (self.list_size) % (self.length_of_single_pass)
        else:
            assert self.list_size >= self.batch_size
            assert (
                self.length_of_single_pass >= self.batch_size
            ), "m * (number of unique labels) must be >= batch_size"
            assert (
                self.batch_size % self.m_per_class
            ) == 0, "m_per_class must divide batch_size without any remainder"
            self.list_size -= self.list_size % self.batch_size

    def __len__(self):
        return self.list_size

    def __iter__(self):
        idx_list = np.array([], dtype=np.int64)
        while len(idx_list) < self.list_size:
            common_functions.NUMPY_RANDOM.shuffle(self.labels)
            if self.batch_size is None:
                curr_label_set = self.labels
            else:
                curr_label_set = self.labels[: self.batch_size // self.m_per_class]
            for label in curr_label_set:
                t = self.labels_to_indices[label]
                n_items_remaining = self.list_size - len(idx_list)
                if n_items_remaining == 0:
                    break
                size = min(self.m_per_class, len(t), n_items_remaining)
                items = common_functions.NUMPY_RANDOM.choice(t, size, replace=False)
                idx_list = np.concatenate([idx_list, items], axis=0)
        return iter(idx_list.tolist())
