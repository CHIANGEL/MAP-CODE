import sys
import os
import numpy as np
import torch
from torch import nn
import random


def get_ptBS(ptType, model, dataset_name):
    if ptType == 'MFP':
        ptBS = '4096'
    elif ptType == 'RFD':
        ptBS = '4096'
    elif ptType == 'SCARF':
        if model in ['FiGNN']:
            ptBS = '1024'
        else:
            ptBS = '2048'
    elif ptType == 'MF4UIP':
        if 'criteo' in dataset_name or model in ['trans', 'xDeepFM', 'DNN', 'DCNv2']:
            ptBS = '256'
        else:
            ptBS = '64'
    else:
        assert 0
    return ptBS

def get_load_step(dataset_name, ptEpoch, ptBS):
    if dataset_name == 'avazu_x4':
        if ptBS == '4096':
            load_step = 7897 * int(ptEpoch)
        elif ptBS == '2048':
            load_step = 15793 * int(ptEpoch)
        elif ptBS == '1024':
            load_step = 31586 * int(ptEpoch)
        elif ptBS == '256':
            load_step = 126341 * int(ptEpoch)
        elif ptBS == '64':
            load_step = 505363 * int(ptEpoch)
    elif dataset_name == 'criteo_x4':
        if ptBS == '4096':
            load_step = 8954 * int(ptEpoch)
        elif ptBS == '2048':
            load_step = 17907 * int(ptEpoch)
        elif ptBS == '1024':
            load_step = 35813 * int(ptEpoch)
        elif ptBS == '256':
            load_step = 143252 * int(ptEpoch)
    else:
        assert 0
    return load_step

def get_data_dir(dataset_name):
    if dataset_name == 'avazu_x4':
        data_dir = '/home/chiangel/data/avazu/avazu_x4'
        # data_dir = '../data/avazu/avazu_x4'
    elif dataset_name == 'criteo_x4':
        data_dir = '/home/chiangel/data/criteo/criteo_x4'
        # data_dir = '../../gated-product-unit/data/criteo/criteo_x4'
    else:
        assert 0
    return data_dir


def parse_wandb_name(wandb_name):
    if wandb_name == 'avazu_x4_core2_emb16':
        dataset_name = 'avazu_x4'
        n_core = '2'
        embed_size = 16
    elif wandb_name == 'avazu_x4_core2_emb32':
        dataset_name = 'avazu_x4'
        n_core = '2'
        embed_size = 32
    elif wandb_name == 'avazu_x4_core2_emb64':
        dataset_name = 'avazu_x4'
        n_core = '2'
        embed_size = 64
    elif wandb_name == 'criteo_x4_core10_emb64':
        dataset_name = 'criteo_x4'
        n_core = '10'
        embed_size = 64
    elif wandb_name == 'criteo_x4_core10_emb32':
        dataset_name = 'criteo_x4'
        n_core = '10'
        embed_size = 32
    elif wandb_name == 'criteo_x4_core10_emb16':
        dataset_name = 'criteo_x4'
        n_core = '10'
        embed_size = 16
    elif wandb_name == 'criteo_x4_core10_emb8':
        dataset_name = 'criteo_x4'
        n_core = '10'
        embed_size = 8
    else:
        assert 0
    return dataset_name, n_core, embed_size


def setup_print_for_ddp(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(gpu=-1):
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu))
    else:
        device = torch.device("cpu")
    return device


class cached_property(property):

    def __get__(self, obj, objtype=None):
        # See docs.python.org/3/howto/descriptor.html#properties
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        attr = "__cached_" + self.fget.__name__
        cached = getattr(obj, attr, None)
        if cached is None:
            cached = self.fget(obj)
            setattr(obj, attr, cached)
        return cached
