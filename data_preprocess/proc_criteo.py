import json
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from timeit import default_timer as timer
from tqdm import tqdm
import pickle as pkl
import pandas as pd
import h5py
import math


# data_dir = '/home/chiangel/data/criteo/criteo_x4/'
data_dir = '../data/criteo/criteo_x4/'
feat_names = ['click'] + \
             ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'] + \
             ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', \
              'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']
valid_fields = feat_names
feat_type_dict = {name: '<cat>' if 'C' in name else '<num>' for name in valid_fields if name != 'click'}


def trans_num_feat(df, col_name):
    def _convert_to_bucket(value):
        try:
            if type(value) != type('') and math.isnan(value):
                value = -1
            elif int(value) > 2:
                value = int(np.floor(np.log(int(value)) ** 2))
            else:
                value = int(value)
        except:
            print(col_name)
            print(value)
            print(type(value))
            assert 0
        return value
    return df[col_name].map(_convert_to_bucket).astype(int)


def trans_cat_feat(df, col_name):
    def _filter_empty_feat(value):
        if type(value) != type('') and math.isnan(value):
            value = '-1'
        return value
    return df[col_name].map(_filter_empty_feat)


def read_feat():
    print(f'... reading the whole data file...')
    tic = timer()
    df = pd.read_csv(os.path.join(data_dir, 'dac', 'train.txt'), sep='\t', header=None, names=feat_names, encoding='utf-8', dtype=object)
    print(df.info())
    # print(df.describe())
    toc = timer()
    print('load', toc - tic)
        
    for name in feat_names:
        print(f'... processing field: {name}...')
        if os.path.exists(os.path.join(data_dir, f'{name}.npy')):
            print('npy file exists; continue.')
            continue
        
        # transform
        tic = timer()
        if 'I' in name:
            feat = trans_num_feat(df, name).to_numpy()
        elif 'C' in name:
            feat = trans_cat_feat(df, name).to_numpy()
        else:
            feat = df[name].to_numpy()
        toc = timer()
        print('transform', toc - tic)
        
        # save
        tic = timer()
        np.save(os.path.join(data_dir, f'{name}.npy'), feat)
        del feat
        toc = timer()
        print('save', toc - tic)

        # load
        tic = timer()
        feat = np.load(os.path.join(data_dir, f'{name}.npy'), allow_pickle=True)
        del feat
        toc = timer()
        print('load', toc - tic)


def generate_dataset(n_core=10, down_sample=None):
    print(f'==> generate dataset with n_core={n_core}')
    np.random.seed(42)

    assert os.path.exists(os.path.join(data_dir, 'click.npy'))
    labels = np.load(os.path.join(data_dir, 'click.npy'), allow_pickle=True).astype(np.int64)

    index = np.arange(len(labels))
    if down_sample:
        np.random.shuffle(index)
        index = index[:down_sample]
        print('down sampled indices', index)
    labels = labels[index]
    print('avg ctr', labels.mean())

    # generate field_map, feat_type_map
    print(f'... generate field_map, feat_type_map ...')
    tic = timer()
    feat_map, field_map, feat_type_map = {}, {}, {}
    feat_type_map['<rsv>'] = len(feat_type_map)
    field_map['<rsv>'] = len(field_map)
    feat_map['<pad>'] = len(feat_map)
    feat_map['<cls>'] = len(feat_map)
    feat_map['<sep>'] = len(feat_map)
    feat_map['<mask>'] = len(feat_map)
    for i in range(6):
        feat_map[f'<unused{i}>'] = len(feat_map)
    assert len(feat_map) == 10

    for name in valid_fields:
        if name == 'click':
            continue
        else:
            field_map[name] = len(field_map)
            feat_type = feat_type_dict[name]
            if feat_type not in feat_type_map:
                feat_type_map[feat_type] = len(feat_type_map)
    if '<oov>' not in feat_type_map:
        feat_type_map['<oov>'] = len(feat_type_map)
    print('field_map:')
    print(json.dumps(field_map, indent=2, ensure_ascii=False))
    print('feat_type_map:')
    print(json.dumps(feat_type_map, indent=2, ensure_ascii=False))

    all_feat_ids, all_field_ids, all_type_ids = [], [], []
    for name in valid_fields:
        if name == 'click':
            continue
        else:
            tic = timer()
            assert os.path.exists(os.path.join(data_dir, f'{name}.npy'))
            feat = np.load(os.path.join(data_dir, f'{name}.npy'), allow_pickle=True)
            toc = timer()
            print('load', toc - tic)

            print(f'field: {name}, type: {feat.dtype}')
            feat = feat[index]
            print(feat, feat.shape)
            
            for k, v in tqdm(Counter(feat).most_common()):
                if v >= n_core:
                    feat_map[f'{name}-{k}'] = len(feat_map)
            feat_map[f'{name}-<oov>'] = len(feat_map)

            feat_ids, field_ids, type_ids = [], [], []
            for f in feat:
                key = f'{name}-{f}'
                if key in feat_map:
                    feat_ids.append(feat_map[key])
                else:
                    feat_ids.append(feat_map[f'{name}-<oov>'])
                if feat_type_dict[name] == '<cat>' and f == -1:
                    type_ids.append(feat_type_map['<oov>'])
                if feat_type_dict[name] == '<num>' and f == -1:
                    type_ids.append(feat_type_map['<oov>'])
                else:
                    type_ids.append(feat_type_map[feat_type_dict[name]])
        
        all_feat_ids.append(feat_ids)
        all_field_ids.append(np.ones(len(feat), dtype=np.int32) * field_map[name])
        all_type_ids.append(type_ids)
        

    print(f'feat_map (input_size = {len(feat_map)}):')
    print(json.dumps(feat_map, indent=2, ensure_ascii=False))

    meta_data = {'index': index.tolist(), 'num_pos': int(labels.sum()), 'num_neg': int(len(labels) - labels.sum()),
                 'avg_ctr': float(labels.mean()), 'field_names': ['<rsv>'] + valid_fields[1:], 'field_map': field_map,
                 'feat_type_map': feat_type_map, 'feat_map': feat_map}
    json.dump(meta_data, open(f'../data/criteo/criteo_x4/criteo_x4_{n_core}-core.json', 'w'), ensure_ascii=False)

    all_feat_ids = np.array(all_feat_ids).transpose()
    all_field_ids = np.array(all_field_ids).transpose()
    all_type_ids = np.array(all_type_ids).transpose()
    print(all_feat_ids, all_feat_ids.shape)
    print(all_field_ids, all_field_ids.shape)
    print(all_type_ids, all_type_ids.shape)
    
    with h5py.File(f'../data/criteo/criteo_x4/criteo_x4_{n_core}-core.h5', 'w') as hf:
        hf.create_dataset('feat_ids', data=all_feat_ids)
        hf.create_dataset('field_ids', data=all_field_ids)
        hf.create_dataset('type_ids', data=type_ids)
        hf.create_dataset('labels', data=labels)
    
    with h5py.File(f'../data/criteo/criteo_x4/criteo_x4_{n_core}-core.h5', 'r') as hf:
        feat_ids = hf['feat_ids'][:]
        field_ids = hf['field_ids'][:]
        type_ids = hf['type_ids'][:]
        labels = hf['labels'][:]
    print('feat_ids', feat_ids.shape)
    print('field_ids', field_ids.shape)
    print('type_ids', type_ids.shape)
    print('labels', labels.shape)


if __name__ == "__main__":
    # read_feat()
    
    n_core = int(sys.argv[1])
    generate_dataset(n_core=n_core, down_sample=None)

    meta_data = json.load(open(f'../data/criteo/criteo_x4/criteo_x4_{n_core}-core.json', 'r'))
    for k, v in meta_data.items():
        if isinstance(v, dict):
            print(k, 'first 10 examples')
            for i, (_k, _v) in enumerate(v.items()):
                print(f'{_k}:\t{_v}')
                if i == 10:
                    break
        elif isinstance(v, list):
            print(k, np.array(v))
        else:
            print(k, v)
