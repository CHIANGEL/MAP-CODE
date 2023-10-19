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

# from proc_data import *

data_dir = '../data/avazu/'
raw_train_file = 'train.gz'
raw_test_file = 'test.gz'
feat_names = ['click', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain',
              'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15',
              'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
valid_fields = ['click', 'weekday', 'day', 'hour', 'is_weekend', 'C1', 'banner_pos', 'site_id', 'site_domain',
                'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model',
                'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
feat_type_dict = {name: '<cat>' for name in valid_fields if name != 'click'}


def read_feat():
    # df = read_gz_csv(data_dir + raw_train_file)
    # df = df.drop(columns=['id'])

    for name in feat_names:
        if os.path.exists(data_dir + f'{name}.npy'):
            continue

        # name = 'hour'
        print(f'... processing field: {name}...')

        tic = timer()
        df = pd.read_csv(data_dir + raw_train_file, usecols=[name], compression='gzip', sep=',')
        df.info()
        df.describe(datetime_is_numeric=True)
        toc = timer()
        print('read', toc - tic)

        # extract weekday, day_of_month, hour_of_day, is_weekend from hour
        if name == 'hour':
            feat = pd.to_datetime(df[name], format='%y%m%d%H')

            weekday, day, hour, is_weekend = [], [], [], []
            for date in tqdm(feat):
                weekday.append(date.weekday())
                day.append(date.day)
                hour.append(date.hour)
                is_weekend.append(int(weekday[-1] > 4))
            weekday, day, hour, is_weekend = np.array(weekday), np.array(day), np.array(hour), np.array(is_weekend)

            tic = timer()
            np.save(data_dir + 'weekday.npy', weekday)
            np.save(data_dir + 'day.npy', day)
            np.save(data_dir + 'hour.npy', hour)
            np.save(data_dir + 'is_weekend.npy', is_weekend)
            toc = timer()
            print('save', toc - tic)

            tic = timer()
            np.load(data_dir + 'weekday.npy', allow_pickle=True)
            np.load(data_dir + 'day.npy', allow_pickle=True)
            np.load(data_dir + 'hour.npy', allow_pickle=True)
            np.load(data_dir + 'is_weekend.npy', allow_pickle=True)
            toc = timer()
            print('load', toc - tic)
            continue
            # exit(0)

        tic = timer()
        feat = df[name].to_numpy()
        np.save(data_dir + f'{name}.npy', feat)
        del df, feat
        toc = timer()
        print('save', toc - tic)

        tic = timer()
        feat = np.load(data_dir + f'{name}.npy', allow_pickle=True)
        print(feat)
        toc = timer()
        print('load', toc - tic)


def plot_hist(data, label=None, logy=True):
    if isinstance(data[0], str):
        uniq_vals = np.unique(data)
        if len(uniq_vals) > 100:
            bins = 100
        else:
            bins = 10
        heights, _, _ = plt.hist(data, bins=bins)
        if np.max(heights) > 1000 and logy:
            plt.yscale('log')
            plt.ylabel('log histogram')
        else:
            plt.ylabel('histogram')
        plt.xticks([], [])
        plt.xlabel(f'{label}, cat({len(uniq_vals)})')
        plt.savefig(data_dir + f'fig/{label}.pdf')
        plt.show()
    else:
        count = []
        for k, v in Counter(data).most_common():
            count.append([k, v])
        count = np.array(count)
        width = count[:, 0].max() - count[:, 0].min()
        width = max(min(1, width / 10), width / 50)
        plt.bar(count[:, 0], count[:, 1], width=width)
        if count[:, 1].max() > 1000 and logy:
            plt.yscale('log')
            plt.ylabel('log histogram')
        else:
            plt.ylabel('histogram')
        plt.xlabel(f'{label}, val({len(count)}), range({count[:, 0].min()}, {count[:, 0].max()})')
        plt.savefig(data_dir + f'fig/{label}.pdf')
        plt.show()


def stat(data):
    counter, miss = [], '-'
    for k, v in tqdm(Counter(data).most_common()):
        counter.append([k, v])
        if k == -1:
            miss = v
    counter = np.array(counter)

    if isinstance(data[0], str):
        min_val, max_val = '-', '-'
        mean, std = '-', '-'
    else:
        min_val, max_val = data.min(), data.max()
        mean, std = data.mean(), data.std()
    unique, total = len(counter), len(data)
    top, top_freq = counter[0]
    low, low_freq = counter[-1]
    try:
        miss_rate = miss / total
    except TypeError:
        miss_rate = '-'
    return counter, [min_val, max_val, mean, std, unique, total, top, top_freq, low, low_freq, miss, miss_rate]


def basic_stat(show_plot=False, n_cores=(1,2, 5, 10, 15, 20)):
    all_stat = []
    columns = ['min', 'max', 'mean', 'std', 'unique', 'total', 'top', 'top_freq', 'low', 'low_freq', 'miss', 'miss_rate']
    all_counters = []
    for name in valid_fields:
        # name = 'is_weekend'
        assert os.path.exists(data_dir + f'{name}.npy')
        feat = np.load(data_dir + f'{name}.npy', allow_pickle=True)
        print(f'field: {name}, type: {feat.dtype}')
        print(feat)

        if show_plot:
            plot_hist(feat, label=name)

        counter, single_stat = stat(feat)
        print(counter)
        all_stat.append(single_stat)
        all_counters.append(counter)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        df = pd.DataFrame(all_stat, index=valid_fields, columns=columns)
        print(df)

    n_core_stats = []
    field_sizes = []
    for counter in all_counters:
        row = []
        row_sizes = []
        for n_core in n_cores:
            drop, field_size = 0, len(counter) + 1
            for k, v in counter[::-1]:
                v = int(v)
                if v < n_core:
                    drop += v
                    field_size -= 1
                else:
                    break
            row.append((drop, field_size))
            row_sizes.append(field_size)
        n_core_stats.append(row)
        field_sizes.append(row_sizes)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        columns = [f'{n}-core drops' for n in n_cores]
        field_sizes = np.array(field_sizes).sum(axis=0)
        df = pd.DataFrame(n_core_stats + [field_sizes], index=valid_fields + ['total_dim'], columns=columns)
        print(df)


def generate_dataset(n_core=5, down_sample=None):
    np.random.seed(42)

    assert os.path.exists(data_dir + 'click.npy')
    labels = np.load(data_dir + 'click.npy', allow_pickle=True)

    index = np.arange(len(labels))
    np.random.shuffle(index)
    if down_sample:
        index = index[:down_sample]
        print('down sampled indices', index)
    labels = labels[index]
    print('avg ctr', labels.mean())

    # generate field_map, feat_type_map
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
            assert os.path.exists(data_dir + f'{name}.npy')
            feat = np.load(data_dir + f'{name}.npy', allow_pickle=True)

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
    json.dump(meta_data, open(f'../data/avazu/avazu_x4/avazu_x4_{n_core}-core.json', 'w'), ensure_ascii=False)

    all_feat_ids = np.array(all_feat_ids).transpose()
    all_field_ids = np.array(all_field_ids).transpose()
    all_type_ids = np.array(all_type_ids).transpose()
    print(all_feat_ids, all_feat_ids.shape)
    print(all_field_ids, all_field_ids.shape)
    print(all_type_ids, all_type_ids.shape)
    
    with h5py.File(f'../data/avazu/avazu_x4/avazu_x4_{n_core}-core.h5', 'w') as hf:
        hf.create_dataset('feat_ids', data=all_feat_ids)
        hf.create_dataset('field_ids', data=all_field_ids)
        hf.create_dataset('type_ids', data=type_ids)
        hf.create_dataset('labels', data=labels)
    
    with h5py.File(f'../data/avazu/avazu_x4/avazu_x4_{n_core}-core.h5', 'r') as hf:
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
    # basic_stat()
    n_core = int(sys.argv[1])
    generate_dataset(n_core=n_core, down_sample=None)

    # meta_data = json.load(open('../data/avazu/avazu_x4/avazu_x4_5-core.json', 'r'))
    # for k, v in meta_data.items():
    #     if isinstance(v, dict):
    #         print(k, 'first 10 examples')
    #         for i, (_k, _v) in enumerate(v.items()):
    #             print(f'{_k}:\t{_v}')
    #             if i == 10:
    #                 break
    #     elif isinstance(v, list):
    #         print(k, np.array(v))
    #     else:
    #         print(k, v)
