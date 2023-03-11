import json
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import argparse
import keras_tuner as kt
import tensorflow as tf
import numpy as np
import random
from Utils import RANDOM_STATE, get_dl_dataset, segment_length_split, get_fre_list, resample
from Hyper_deep_models import GRUHyperModel, LSTM1DHyperModel,\
    CNNLSTM2DHyperModel, BiLSTMHyperModel, CNN1DHyperModel
import time

parser = argparse.ArgumentParser(description='Run tuning experiment.')
parser.add_argument('-m', '--model', type=str, help='model name: cnn_1d, lstm_1d, bi_lstm, cnn_lstm, GRU')
parser.add_argument('-e', '--epochs', type=int, help='int, number of epoch')
parser.add_argument('-t', '--max_trials', type=int, help='int, number of max trials of tuning')
parser.add_argument('-k', '--segment_split', default=1, type=float, help='window size is 1024/k')
parser.add_argument('-norm', '--normalize_gait', default=0, type=int, help='0 not; 1 normalize gait')

# set the random state
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# tun the deep learning models, for the HPC
# multi GPU version
if __name__ == '__main__':

    args = parser.parse_args()
    model_name = args.model
    k = args.segment_split
    max_trials = args.max_trials
    m_epoch = args.epochs
    norm_gait = args.normalize_gait

    # if merge the segments, do not shuffle the order
    if k < 1:
        shuffle_flag = 0
    else:
        shuffle_flag = 1
    # load dataset
    X_train, X_val, X_test, y_train, y_val, y_test = get_dl_dataset(0.2, 0.1, shuffle_flag)

    if norm_gait == 1:
        print('==========normalize gait ========')
        # get frequency for each segments
        fre_train = get_fre_list(X_train)
        fre_val = get_fre_list(X_val)
        fre_test = get_fre_list(X_test)

    # split or merge the segments
    X_train, X_val, X_test, y_train, y_val, y_test = segment_length_split(X_train, X_val, X_test, y_train, y_val, y_test, k, shuffle_flag)

    if norm_gait == 1:
        X_train = resample(fre_train, X_train, k)
        X_val = resample(fre_val, X_val, k)
        X_test = resample(fre_test, X_test, k)

    input_shape = X_train.shape[1:]

    # hyper model
    classifiers = {
        'cnn_1d': CNN1DHyperModel(input_shape=(input_shape)),
        'lstm_1d': LSTM1DHyperModel(input_shape=input_shape),
        'bilstm': BiLSTMHyperModel(input_shape=input_shape),
        'cnn_lstm': CNNLSTM2DHyperModel((input_shape[0], input_shape[1], 1)),
        'GRU': GRUHyperModel(input_shape=input_shape),
    }
    # set model
    hp_model = classifiers[model_name]

    objective_metric = 'val_loss'

    print(f'========================================{model_name}========================================')

    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)

    # # model tuning
    tuner = kt.BayesianOptimization(
        hypermodel=hp_model.build_model,
        objective=objective_metric,
        max_trials=max_trials,
        seed=RANDOM_STATE,
        executions_per_trial=3,
        overwrite=True,
        project_name=model_name,
        distribution_strategy=strategy,
        directory=f'./tuning_log/{int(round(time.time()*1000))}'
    )

    if model_name == 'cnn_lstm':
        X_train = np.expand_dims(X_train.reshape(X_train.shape[0], X_train.shape[1], 3, 1), 1)
        y_train = np.expand_dims(y_train, 1)
        X_test = np.expand_dims(X_test.reshape(X_test.shape[0], X_test.shape[1], 3, 1), 1)
        X_val = np.expand_dims(X_val.reshape(X_val.shape[0], X_val.shape[1], 3, 1), 1)
        y_val = np.expand_dims(y_val, 1)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor=objective_metric, patience=20)

    tuner.search(X_train, y_train, epochs=m_epoch, validation_data=(X_val, y_val),
                 callbacks=[stop_early], verbose=2, use_multiprocessing=True)

    best_model = tuner.get_best_models(1)[0]

    # val_loss, val_acc = best_model.evaluate(X_val, y_val)
    best_hps = tuner.get_best_hyperparameters(1)[0]
    print(best_model.summary())
    print(f'{model_name} best parameters is {best_hps.values}')

    # evaluation
    from sklearn import metrics

    pred_probability = best_model.predict(X_test)

    if model_name == 'cnn_lstm' or model_name == 'bi_lstm':
        pred_class = np.argmax(pred_probability, axis=2)
        pred_probability_ = pred_probability[:, :, 1]
    else:
        pred_class = np.argmax(pred_probability, axis=1)
        pred_probability_ = pred_probability[:, 1]

    true_class = np.argmax(y_test, axis=1)

    pre = metrics.precision_score(true_class, pred_class)
    rec = metrics.recall_score(true_class, pred_class)
    f1 = metrics.f1_score(true_class, pred_class)
    acc = metrics.accuracy_score(true_class, pred_class)
    print(f'========================================{model_name}======================================')
    print(f'k is {k}')
    print(f'Precision: {pre}')
    print(f'Recall: {rec}')
    print(f'f1_score: {f1}')
    print(f'Testing Accuracy: {acc}')

    fpr, tpr, thresholds = metrics.roc_curve(true_class, pred_probability_)
    auc = metrics.auc(fpr, tpr)
    print(f'AUC : {auc}')

    current_time = time.strftime("%m_%d_%H_%M")
    print(f'current_time is {current_time}')
    best_model.save(f'./result/models/{model_name}_{k}_{current_time}.h5')
