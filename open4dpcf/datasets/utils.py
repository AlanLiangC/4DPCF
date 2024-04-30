from .common import CollateFn
import torch.utils.data
from timm.data.distributed_sampler import OrderedDistributedSampler
import random
import numpy as np
from functools import partial
from typing import Callable


def worker_init(worker_id, worker_seeding='all'):
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info.id == worker_id
    if isinstance(worker_seeding, Callable):
        seed = worker_seeding(worker_info)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
    else:
        assert worker_seeding in ('all', 'part')
        # random / torch seed already called in dataloader iter class w/ worker_info.seed
        # to reproduce some old results (same seed + hparam combo), partial seeding
        # is required (skip numpy re-seed)
        if worker_seeding == 'all':
            np.random.seed(worker_info.seed % (2 ** 32 - 1))


def create_loader(dataset, shuffle, is_training, drop_last, worker_seeding='all', **kwards):

    sampler = None
    distributed = kwards['distributed']
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)

    loader_class = torch.utils.data.DataLoader

    loader_args = dict(
        batch_size=kwards['batch_size'],
        shuffle=shuffle and (not isinstance(dataset, torch.utils.data.IterableDataset)) and sampler is None and is_training,
        num_workers=kwards['num_workers'],
        sampler=sampler,
        collate_fn=CollateFn,
        pin_memory=kwards['pin_memory'],
        drop_last=drop_last,
        worker_init_fn=partial(worker_init, worker_seeding=worker_seeding),
        persistent_workers=True
    )
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)
    return loader