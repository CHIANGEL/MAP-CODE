import os, sys
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict
import json
import h5py
from collections import Counter


class BaseDataset:
    def __init__(self, args):
        self.args = args
        self.data_dir = args.data_dir
        self.dataset_name = args.dataset_name
        self.split_names = ["train", "valid", "test"]
        self.load_data()
    
    def load_data(self):
        # Load meta data
        meta_data = json.load(open(os.path.join(self.data_dir, f"{self.dataset_name}-meta.json"), "r"))
        self.field_names, self.feat_map, self.field_map = \
            meta_data["field_names"], meta_data["feat_map"], meta_data["field_map"]
        
        # Load CTR data
        with h5py.File(os.path.join(self.data_dir, f"{self.dataset_name}.h5"), "r") as f:
            feat_ids = f["feat_ids"][:]
            labels = f["labels"][:]
        
        # Load split index
        split_index = pkl.load(open(os.path.join(self.data_dir, "split.pkl"), "rb"))
        split_indices = {split: split_index[f"{split}_index"] for split in self.split_names}
        
        # Split the CTR data
        self.X = {split: feat_ids[split_indices[split]] for split in self.split_names}
        self.Y = {split: labels[split_indices[split]] for split in self.split_names}

        self.get_feat_count_file()
        self.count_feat_per_field(feat_ids)
        
    def get_splited_dataset(self, split):
        assert split in self.split_names, f"Unsupported split name: {split}"
        return OurDataset(
            self.X[split], 
            self.Y[split], 
        )
    
    def get_feat_count_file(self):
        feat_count_file = os.path.join(self.data_dir, f"feat-count.pt")
        if self.args.pretrain:
            if os.path.exists(feat_count_file):
                self.feat_count = torch.load(feat_count_file)
            else:
                self.feat_count = torch.zeros(len(self.feat_map))
                feat_list = self.X["train"].flatten().tolist()
                feat_count_dict = Counter(feat_list)
                for i in range(self.feat_count.shape[0]):
                    self.feat_count[i] = feat_count_dict[i]
                torch.save(self.feat_count, feat_count_file)
        else:
            self.feat_count = None
    
    def count_feat_per_field(self, feat_ids):
        if self.args.pt_type == "RFD" and self.args.RFD_replace == "Uniform":
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


class OurDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, k):
        return self.X[k], self.Y[k]
