"""Compute PCA features over the summary metrics from previous script"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

test = pd.read_pickle('./data/test_tab_feats.pkl')
train = pd.read_pickle('./data/train_tab_feats.pkl')

test_fnames = test.fname.values.copy()
test = test.drop(columns='fname').copy()
train_fnames = train.fname.values.copy()
train = train.drop(columns='fname').copy()

X_train = train.values
X_test = test.values

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

exp_var = pd.DataFrame({'exp_var': pca.explained_variance_ratio_})
exp_var['var_cumulative'] = exp_var.exp_var.cumsum()
exp_var.head()

print('Number of PCA features to account for 90% variance: ', (exp_var.var_cumulative <= 0.9).sum())
print('Number of PCA features to account for 95% variance: ', (exp_var.var_cumulative <= 0.95).sum())
print('Number of PCA features to account for 96% variance: ', (exp_var.var_cumulative <= 0.96).sum())
print('Number of PCA features to account for 97% variance: ', (exp_var.var_cumulative <= 0.97).sum())
print('Number of PCA features to account for 98% variance: ', (exp_var.var_cumulative <= 0.98).sum())
print('Number of PCA features to account for 99% variance: ', (exp_var.var_cumulative <= 0.99).sum())

np.save('./data/train_pca_feats.npy', X_train)
np.save('./data/test_pca_feats.npy', X_test)
