import tensorflow as tf
import numpy as np
import random
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn import metrics
from Utils import get_dl_dataset, plot_cm, plot_roc, RANDOM_STATE, plot_roc_compare, \
    segment_length_split, resample, get_fre_list
import os
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# evaluate all the fine-tuning deep learning models
# get all accuracy metric and plot roc and confusion matrix figures
if __name__ == '__main__':
    k = 1  # set segment length: k=1 -> 1024 samples; k=2 ->512; k=4 -> 256; k=8 ->128
    normalized_gait = False  # normalized gait frequency or not

    # if merge the segments, do not shuffle the order
    if k < 1:
        shuffle_flag = 0
    else:
        shuffle_flag = 1

    # load dataset
    X_train, X_val, X_test, y_train, y_val, y_test = get_dl_dataset(0.2, 0.1, shuffle_flag)
    if normalized_gait:
        print('==========normalize gait ========')
        # get frequency for each segments
        fre_train = get_fre_list(X_train)
        fre_val = get_fre_list(X_val)
        fre_test = get_fre_list(X_test)

    X_train, X_val, X_test, y_train, y_val, y_test = segment_length_split(X_train, X_val, X_test, y_train, y_val, y_test, k, shuffle_flag)

    if normalized_gait:
        model_path = 'normalized'
        X_train = resample(fre_train, X_train, k)
        X_val = resample(fre_val, X_val, k)
        X_test = resample(fre_test, X_test, k)
    else:
        model_path = 'raw'

    input_shape = X_train.shape[1:]

    plot_save_path = f'./result/model_plots'

    # models
    dl_model = ['cnn_1d', 'cnn_lstm', 'GRU', 'bilstm', 'lstm_1d']

    fpr_list, tpr_list, auc_list, model_name_list = [], [], [], []

    for model_name in dl_model:
        # prepare the data
        if model_name == 'cnn_lstm':
            X_train_ = np.expand_dims(X_train.reshape(X_train.shape[0], X_train.shape[1], 3, 1), 1)
            y_train_ = np.expand_dims(y_train, 1)
            X_test_ = np.expand_dims(X_test.reshape(X_test.shape[0], X_test.shape[1], 3, 1), 1)
            y_test_ = np.expand_dims(y_test, 1)
            X_val_ = np.expand_dims(X_val.reshape(X_val.shape[0], X_val.shape[1], 3, 1), 1)
            y_val_ = np.expand_dims(y_val, 1)
        else:
            X_train_ = X_train
            y_train_ = y_train
            X_test_ = X_test
            y_test_ = y_test
            X_val_ = X_val
            y_val_ = y_val
        # if model is not there, train the model
        if os.path.exists(f'./result/models/{model_path}/{model_name}_{k}.h5'):
            model = keras.models.load_model(f'./result/models/{model_path}/{model_name}_{k}.h5')
            print(f'{model_name} loading')
        else:
            break

        # evaluation
        pred_probability = model.predict(X_test_)
        if model_name == 'cnn_lstm' or model_name == 'bi_lstm':
            pred_class = np.argmax(pred_probability, axis=2)
            pred_probability_ = pred_probability[:,:,1]
        else:
            pred_class = np.argmax(pred_probability, axis=1)
            pred_probability_ = pred_probability[:,1]

        true_class = np.argmax(y_test, axis=1)

        pre = metrics.precision_score(true_class, pred_class)
        rec = metrics.recall_score(true_class, pred_class)
        f1 = metrics.f1_score(true_class, pred_class)
        acc = metrics.accuracy_score(true_class, pred_class)
        print(f'Precision: {pre}')
        print(f'Recall: {rec}')
        print(f'f1_score: {f1}')
        print(f'Testing Accuracy: {acc}')

        # plot cm
        cm = metrics.confusion_matrix(true_class, pred_class)
        label = ['Adult', 'Older adult']
        plot_cm(cm, label, '') # f'Confusion Matrix {model_name}_{model_path} '
        plt.savefig(f'{plot_save_path}/Confusion Matrix {model_name}_{k}_{model_path}.pdf', format='pdf')
        plt.show()

        # plot roc
        fpr, tpr, thresholds = metrics.roc_curve(true_class, pred_probability_)
        auc = plot_roc(fpr, tpr, f'The ROC and AUC for {model_name}')
        plt.savefig(f'{plot_save_path}/ROC_{model_name}_{k}.pdf', format='pdf')
        plt.show()

        print(f'AUC: {auc}')
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc)
        model_name_list.append(model_name)
    # compare roc of different models in one fig
    plot_roc_compare(fpr_list, tpr_list, model_name_list, auc_list, 'ROC comparison of deep learning')
    plt.savefig(f'{plot_save_path}/ROC comparison of deep learning_{k}.pdf', format='pdf')
    plt.show()

