import numpy as np
import pandas as pd
import hashlib
from sklearn.model_selection import StratifiedKFold
import pickle as pkl
import os

"""
NOTICE: We found that even though we fix the random seed, the resulting data split can be different 
due to the potential StratifiedKFold API change in different scikit-learn versions. For 
reproduciblity, `sklearn==0.19.1` is required. We use the python environement by installing 
`Anaconda3-5.2.0-Linux-x86_64.sh`.
"""

RANDOM_SEED = 2018 # Fix seed for reproduction
cols = ['Label']
for i in range(1, 14):
    cols.append('I' + str(i))
for i in range(1, 27):
    cols.append('C' + str(i))

root = '../data/criteo/criteo_x4/'

ddf = pd.read_csv(os.path.join(root, 'dac', 'train.txt'), sep='\t', header=None, names=cols, encoding='utf-8', dtype=object)
X = ddf.values
y = ddf['Label'].map(lambda x: float(x)).values
print(str(len(X)) + ' lines in total')

folds = StratifiedKFold(n_splits=10, shuffle=True,
                        random_state=RANDOM_SEED).split(X, y)

fold_indexes = []
for train_id, valid_id in folds:
    fold_indexes.append(valid_id)
test_index = fold_indexes[0]
valid_index = fold_indexes[1]
train_index = np.concatenate(fold_indexes[2:])

print('criteo')
print(test_index.shape, type(test_index))
print(valid_index.shape, type(valid_index))
print(train_index.shape, type(train_index))
split_index = {
    'train_index': train_index,
    'valid_index': valid_index,
    'test_index': test_index,
}
pkl.dump(split_index, open(os.path.join(root, 'split_x4.pkl'), 'wb'))

criteo_split_index = pkl.load(open(os.path.join(root, 'split_x4.pkl'), 'rb'))
criteo_train_index, criteo_valid_index, criteo_test_index = \
    criteo_split_index['train_index'], criteo_split_index['valid_index'], criteo_split_index['test_index']
assert (train_index - criteo_train_index).sum() == 0
assert (valid_index - criteo_valid_index).sum() == 0
assert (test_index - criteo_test_index).sum() == 0


avazu_split_index = pkl.load(open(os.path.join('../data/avazu/avazu_x4/', 'split_x4.pkl'), 'rb'))
avazu_train_index, avazu_valid_index, avazu_test_index = \
    avazu_split_index['train_index'], avazu_split_index['valid_index'], avazu_split_index['test_index']
print('avazu')
print(avazu_test_index.shape, type(avazu_test_index))
print(avazu_valid_index.shape, type(avazu_valid_index))
print(avazu_train_index.shape, type(avazu_train_index))

test_df = ddf.loc[test_index, :]
test_df.to_csv(os.path.join(root, 'test.csv'), index=False, encoding='utf-8')
valid_df = ddf.loc[valid_index, :]
valid_df.to_csv(os.path.join(root, 'valid.csv'), index=False, encoding='utf-8')
ddf.loc[train_index, :].to_csv(os.path.join(root, 'train.csv'), index=False, encoding='utf-8')

print('Train lines:', len(train_index))
print('Validation lines:', len(valid_index))
print('Test lines:', len(test_index))
print('Postive ratio:', np.sum(y) / len(y))

# Check md5sum for correctness
assert("4a53bb7cbc0e4ee25f9d6a73ed824b1a" == hashlib.md5(open(os.path.join(root, 'train.csv'), 'r').read().encode('utf-8')).hexdigest())
assert("fba5428b22895016e790e2dec623cb56" == hashlib.md5(open(os.path.join(root, 'valid.csv'), 'r').read().encode('utf-8')).hexdigest())
assert("cfc37da0d75c4d2d8778e76997df2976" == hashlib.md5(open(os.path.join(root, 'test.csv'), 'r').read().encode('utf-8')).hexdigest())

print("Reproducing data succeeded!")