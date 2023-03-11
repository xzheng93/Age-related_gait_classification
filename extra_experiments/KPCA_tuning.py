import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import random
from Utils import RANDOM_STATE, get_train_val_tes_index
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.metrics import mean_squared_error
from joblib import dump, load

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

def my_scorer(estimator, X, y=None):
    X_reduced = estimator.transform(X)
    X_preimage = estimator.inverse_transform(X_reduced)
    return -1 * mean_squared_error(X, X_preimage)
'''
tune the KPCA for ml 
'''
if __name__ == '__main__':
    path = '../data/gait_outcomes.csv'
    train_index, val_index, test_index, del_index, labels = get_train_val_tes_index(0.2, 0.1, path)

    df = pd.read_csv(path)
    df = df.drop(del_index, axis=0)
    data = df.to_numpy()
    # delete the id, age, and other duplicate features
    features = np.delete(data, [0, 1, 21, 22, 23, 31, 32, 33, 38, 39, 40, 47, 48, 49], axis=1)
    features = preprocessing.scale(features)

    # KPCA tuning
    param_grid = [{
        'gamma': np.linspace(0.1, 0.01, 5),
        'n_components': list(np.arange(2, 18))
    }]

    kpca = KernelPCA(kernel='rbf',fit_inverse_transform=True, n_jobs=-1)
    grid_search = RandomizedSearchCV(kpca, param_grid, cv=5,
                                     scoring=my_scorer,
                                     n_iter=100,
                                     verbose=2,
                                     random_state=RANDOM_STATE)
    grid_search.fit(features)
    print(grid_search.best_params_)
