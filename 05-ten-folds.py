"""Divide the training data into 10 (stratified by label) folds"""

import pickle
import pandas as pd
from sklearn.model_selection import StratifiedKFold


traindf = pd.read_csv('./data/train.csv')

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=20180629)
folds = list(skf.split(traindf.fname.values, traindf.label.values))

folds_fnames = [(traindf.fname.values[x[0]],
                 traindf.fname.values[x[1]])
                for x in folds]

with open('./data/folds.pkl', 'wb') as handle:
    pickle.dump(folds_fnames, handle)
