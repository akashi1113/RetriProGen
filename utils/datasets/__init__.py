import pickle
import torch
import os
import numpy as np
from torch.utils.data import Subset
from .pl import PocketLigandPairDataset
from .resgen import ResGenDataset


def get_dataset(config, *args, **kwargs):
    name = config.dataset.name
    root = config.dataset.path
    if name == 'pl':
        dataset = PocketLigandPairDataset(config, root, *args, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    if 'split' in config.dataset:
        split_by_name = torch.load(config.dataset.split)
        split = {
            k: [dataset.name2id[n] for n in names if n in dataset.name2id]
            for k, names in split_by_name.items()
        }
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return subsets


def transform_data(data, transform):
    assert data.protein_pos.size(0) > 0
    if transform is not None:
        data = transform(data)
    return data
