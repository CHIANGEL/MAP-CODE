import os, sys
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict
import json
import h5py
from collections import Counter


class DatasetV1(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, item):
        return self.X[item], self.Y[item]


class DataCollatorV1:
    @classmethod
    def collate_batch(cls, features) -> Dict[str, torch.Tensor]:
        feats, labels = [], []
        for f, l in features:
            feats.append(f)
            labels.append(l)
        labels = torch.tensor(labels, dtype=torch.float)
        feats = torch.tensor(np.array(feats), dtype=torch.long)
        return {'input_ids': feats, 'labels': labels}


class DatasetV2(Dataset):
    def __init__(self, X1, X2, X3, Y):
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, item):
        return self.X1[item], self.X2[item], self.X3[item], self.Y[item]


class DataCollatorV2:
    @classmethod
    def collate_batch(cls, features) -> Dict[str, torch.Tensor]:
        input_ids, field_ids, type_ids, labels = [], [], [], []
        for x1, x2, x3, l in features:
            input_ids.append(x1)
            field_ids.append(x2)
            type_ids.append(x3)
            labels.append(l)
        labels = torch.tensor(labels, dtype=torch.float)
        input_ids = torch.tensor(np.array(input_ids), dtype=torch.long)
        field_ids = torch.tensor(np.array(field_ids), dtype=torch.long)
        type_ids = torch.tensor(np.array(type_ids), dtype=torch.long)
        return {'input_ids': input_ids, 'field_ids': field_ids, 'type_ids': type_ids, 'labels': labels}


class BaseDataset:
    ...

    @classmethod
    def get_dataset(cls, args):
        if args.dataset_name.lower() == 'avazu_demo':
            return AvazuDemo()
        elif args.dataset_name.lower() == 'avazu_x4':
            return Avazu_x4(args)
        elif args.dataset_name.lower() == 'criteo_x4':
            return Criteo_x4(args)
        else:
            raise NotImplementedError
    
    def get_meta_data(self):
        meta_data = json.load(open(os.path.join(self.data_dir, self.metadata_name), 'r'))
        self.field_names, self.feat_map, self.field_map, self.feat_type_map = \
            meta_data['field_names'], meta_data['feat_map'], meta_data['field_map'], meta_data['feat_type_map']

    def get_dataset_v1(self, set_name='train'):
        if set_name == 'train':
            return DatasetV1(self.train_X1, self.train_Y)
        elif set_name == 'valid':
            return DatasetV1(self.valid_X1, self.valid_Y)
        elif set_name == 'test':
            return DatasetV1(self.test_X1, self.test_Y)

    def get_dataset_v2(self, set_name='train'):
        if set_name == 'train':
            return DatasetV2(self.train_X1, self.train_X2, self.train_X3, self.train_Y)
        elif set_name == 'valid':
            return DatasetV2(self.valid_X1, self.valid_X2, self.valid_X3, self.valid_Y)
        elif set_name == 'test':
            return DatasetV2(self.test_X1, self.test_X2, self.test_X3, self.test_Y)
    
    def get_feat_count_file(self):
        feat_count_file = os.path.join(self.data_dir, f'feat-count_{self.args.split_name}_{self.args.n_core}-core.pt')
        if self.args.pretrain:
            if os.path.exists(feat_count_file):
                self.feat_count = torch.load(feat_count_file)
            else:
                self.feat_count = torch.zeros(len(meta_data['feat_map']))
                feat_list = self.train_X1.flatten().tolist()
                feat_count_dict = Counter(feat_list)
                for i in range(self.feat_count.shape[0]):
                    self.feat_count[i] = feat_count_dict[i]
                torch.save(self.feat_count, feat_count_file)
        else:
            self.feat_count = None
    
    def count_feat_per_field(self, feat_ids):
        if self.args.pt_type =='MF4UIP' or (self.args.pt_type == 'RFD' and self.args.RFD_G == 'Uniform'):
            # We must use feat_ids to count the feature! 
            # There exists features in the validation set that never apperar in the training set
            self.idx_low = torch.from_numpy(feat_ids.min(axis=0))
            self.idx_high = torch.from_numpy(feat_ids.max(axis=0) + 1)
            self.feat_num_per_field = self.idx_high - self.idx_low
            assert (self.feat_num_per_field - (torch.from_numpy(feat_ids) - self.idx_low.view(1, -1)) <= 0).sum() == 0
        else:
            self.idx_low = None
            self.idx_high = None
            self.feat_num_per_field = None


class Avazu_x4(BaseDataset):
    def __init__(self, args):
        super(Avazu_x4, self).__init__()
        self.args = args
        self.data_dir = args.data_dir
        self.dataset_name = f'avazu_x4_{args.n_core}-core'
        self.metadata_name = f'avazu_x4_{args.n_core}-core.json'

        with h5py.File(os.path.join(self.data_dir, f'{self.dataset_name}.h5'), 'r') as f:
            feat_ids = f['feat_ids'][:]
            # field_ids = f['field_ids'][:]
            # type_ids = f['type_ids'][:]
            labels = f['labels'][:]

        # avazu x4的index
        if args.split_name == 'x4-full':
            split_index = pkl.load(open(os.path.join(self.data_dir, 'split_x4.pkl'), 'rb'))
        else:
            raise NotImplementedError
        train_index, valid_index, test_index = split_index['train_index'], split_index['valid_index'], split_index['test_index']
        self.train_X1, self.train_X2, self.train_X3 = feat_ids[train_index], None, None
        self.valid_X1, self.valid_X2, self.valid_X3 = feat_ids[valid_index], None, None
        self.test_X1, self.test_X2, self.test_X3 = feat_ids[test_index], None, None
        self.train_Y, self.valid_Y, self.test_Y = labels[train_index], labels[valid_index], labels[test_index]

        self.get_meta_data()
        self.get_feat_count_file()
        self.count_feat_per_field(feat_ids)


class Criteo_x4(BaseDataset):
    def __init__(self, args):
        super(Criteo_x4, self).__init__()
        self.args = args
        self.data_dir = args.data_dir
        self.dataset_name = f'criteo_x4_{args.n_core}-core'
        self.metadata_name = f'criteo_x4_{args.n_core}-core.json'

        with h5py.File(os.path.join(self.data_dir, f'{self.dataset_name}.h5'), 'r') as f:
            feat_ids = f['feat_ids'][:]
            # field_ids = f['field_ids'][:]
            # type_ids = f['type_ids'][:]
            labels = f['labels'][:]

        # criteo x4的index
        if args.split_name == 'x4-full':
            split_index = pkl.load(open(os.path.join(self.data_dir, 'split_x4.pkl'), 'rb'))
        elif args.split_name == 'pt':
            split_index = pkl.load(open(os.path.join(self.data_dir, 'split_pt.pkl'), 'rb'))
        elif args.split_name == 'ch':
            split_index = pkl.load(open(os.path.join(self.data_dir, 'split_ch.pkl'), 'rb'))
        else:
            raise NotImplementedError
        train_index, valid_index, test_index = split_index['train_index'], split_index['valid_index'], split_index['test_index']
        self.train_X1, self.train_X2, self.train_X3 = feat_ids[train_index], None, None
        self.valid_X1, self.valid_X2, self.valid_X3 = feat_ids[valid_index], None, None
        self.test_X1, self.test_X2, self.test_X3 = feat_ids[test_index], None, None
        self.train_Y, self.valid_Y, self.test_Y = labels[train_index], labels[valid_index], labels[test_index]

        self.get_meta_data()
        self.get_feat_count_file()
        self.count_feat_per_field(feat_ids)
