import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import random
from sklearn import metrics
from Utils import get_ml_dataset, RANDOM_STATE
import argparse
import keras_tuner as kt
from sklearn.model_selection import PredefinedSplit
import pickle

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# tun the ML models by BayesianOptimization
# in sklearn_tuner.py change line 169
# from if isinstance(model, sklearn.pipeline.Pipeline):
# to if sample_weight == None:

parser = argparse.ArgumentParser(description='Run tuning experiment for machine learning.')
parser.add_argument('-m', '--model', type=str,help='model name: nb, svc, knn, rf')
parser.add_argument('-t', '--trials', default=15, type=int,help='int, number of trials')

# define the parameter space
def build_rf(hp):
    model = RandomForestClassifier(
        n_estimators=hp.Int('n_estimators', 10, 1000, step=10),
        max_depth=hp.Int('max_depth', 3, 25, step=1),
        max_leaf_nodes=hp.Int('max_leaf_nodes', 5, 50, step=1),
        min_samples_split=hp.Int('min_samples_split', 5, 50, step=1),
    )
    return model


def build_knn(hp):
    model = KNeighborsClassifier(
        n_neighbors=hp.Int('n_estimators', 1, 15, step=1),
        algorithm=hp.Choice(name="algorithm", values=["ball_tree", "kd_tree", "brute"]),
        weights=hp.Choice('weights',['uniform', 'distance']),
    )
    return model


def build_svc(hp):
    model = SVC(
        kernel='rbf',
        C=hp.Int('C', 1, 250, step=1),
        gamma=hp.Float('gamma', 0.01, 10, step=0.05),
        degree=hp.Int('degree', 1, 50, step=1),
        shrinking=hp.Choice('shrinking',[True, False]),
        probability=True
    )
    return model


def build_nb(hp):
    model = GaussianNB(
        var_smoothing=hp.Float('var_smoothing', 1e-11, 1e-7, step=1e-10),
    )
    return model


if __name__ == '__main__':
    args = parser.parse_args()
    model_name = args.model
    trials = args.trials

    # load dataset
    X_train, X_val, X_test, y_train, y_val, y_test = get_ml_dataset(0.2, 0.1)

    # predefine the training and validation data set
    train_fold = np.ones(len(X_train))*-1
    val_fold = np.zeros(len(X_val))
    X_train_val = np.concatenate((X_train, X_val), axis=0)
    y_train_val = np.concatenate((y_train, y_val), axis=0)
    y_train_val = np.argmax(y_train_val, axis=1)

    ps = PredefinedSplit(np.concatenate((train_fold, val_fold), axis=0))


    classifiers = {
        'nb': build_nb,
        'svc': build_svc,
        'knn': build_knn,
        'rf': build_rf
    }

    hp_model = classifiers[model_name]
    print(f'============================================{model_name}=================================')
    tuner = kt.SklearnTuner(
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective('score', 'max'),
            max_trials=trials),
        hypermodel=hp_model,
        scoring=metrics.make_scorer(metrics.accuracy_score),
        cv=ps,
        directory="experiments",
        project_name=model_name,

    )
    tuner.search(X_train_val, y_train_val)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    # best Hyper parameter
    print(f'{model_name} best parameter ')
    print(best_hps.values)
    # save best model
    best_model = tuner.get_best_models(1)[0]
    with open(f'./result/models/{model_name}.pickle', 'wb') as file:
        pickle.dump(best_model, file)

