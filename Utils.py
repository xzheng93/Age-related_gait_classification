import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.utils import to_categorical
from random import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import KernelPCA
import math
from scipy import interpolate

RANDOM_STATE = 0  # random seed
K_samples = 10  # samples k segments data for each participant
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
csfont = {'fontname':'Times New Roman'}


def load_data(path):
    """
    loads data and labels the groups
    :param path:  acc data path
    :return: a list of acc segment, a list of the subjects id, a list of the labels
    """
    # load data
    data = scio.loadmat(path)
    subjects = data['id'].flatten()
    ages = data['age'].flatten()
    acc_X = data['X']
    acc_Y = data['Y']
    acc_Z = data['Z']

    # set age labels
    groups = np.zeros(len(acc_X))
    # groups[(ages > 35) & (ages <= 65)] = 1
    groups[ages > 65] = 1

    # form data [x,x,x,...] [y,y,y,...] [z,z,z,...]  to [[x,y,z],[x,y,z]...]segments
    segments = []
    for i in range(0, acc_X.shape[0]):
        vectors = []
        for j in range(0, acc_X.shape[1]):
            vectors.append([acc_X[i, j], acc_Y[i, j], acc_Z[i, j]])
        segments.append(vectors)
    segments = np.array(segments)

    return segments, subjects, groups


def my_sample(subjects, my_list, k=10, shuffle_flag=1):
    """
    for elements in my_list, sample k from subjects
    :param subjects: a list of subjects id
    :param my_list: a list of target id
    :param k: sample number
    :return: k index of  each elements in my_list
    """
    data_index = []
    for i in range(0, len(my_list)):
        m_index, = np.where(subjects == my_list[i])
        if len(m_index) >= k:
            if shuffle_flag:
                data_index.extend(np.random.choice(m_index, k))
            else:
                data_index.extend(m_index[0:k])
        else:
            print(f'not enough data {len(m_index)}/{k} for subject {my_list[i]}')

    return np.array(data_index)


def get_train_val_tes_index(test_rate, val_rate, path='./data/gait_outcomes.csv', shuffle_flag=1, random_state=0):
    """
    splits data set by id
    :param test_rate:
    :param val_rate:
    :param path:
    :param random_state:
    :return: the list of train, test, validation id.
    """
    # clean dataset
    bad_data_id = [144, 178, 147, 4]  # these subjects has no more than 10 segments (9, 3, 9, 6);
    # 4 subjects have 10, 3 has 11, a lot of subjects has 12, so we sample 10 segments for each subject

    df = pd.read_csv(path)
    del_index = []
    # remove bad id
    for i in range(0, len(bad_data_id)):
        bad_index = df.index[df.id == bad_data_id[i]]
        del_index.extend(bad_index)
    # remove the nan feature data
    na_index = df.index[df.isnull().T.any()].tolist()
    del_index.extend(na_index)
    # remove all 0 data
    del_index.extend(np.where(df['StrideTimeSamples'] == 0)[0])
    df = df.drop(del_index, axis=0)

    subjects = np.array(df.id)
    ages = np.array(df.age)
    uid, uid_index = np.unique(subjects, return_index=True)

    # set age labels
    groups = np.zeros(len(ages))
    groups[ages > 65] = 1
    labels = to_categorical(groups, 2)

    # split the subjects id to train, validation and test
    trVa_id, test_id, trVa_id_y, test_id_y = train_test_split(uid, groups[uid_index],
                                                              random_state=random_state,
                                                              stratify=groups[uid_index],
                                                              test_size=test_rate)

    train_id, val_id, train_id_y, val_id_y = train_test_split(trVa_id, trVa_id_y,
                                                              random_state=random_state,
                                                              stratify=trVa_id_y,
                                                              test_size=val_rate / (1 - test_rate))
    train_index = my_sample(subjects, train_id, K_samples, shuffle_flag)
    val_index = my_sample(subjects, val_id, K_samples, shuffle_flag)
    test_index = my_sample(subjects, test_id, K_samples, shuffle_flag)

    if shuffle_flag:
        shuffle(test_index)
        shuffle(val_index)
        shuffle(train_index)

    return train_index, val_index, test_index, del_index, labels


# gets the dataset for machine learning
# normalization and KPCA
def get_ml_dataset(test_rate, val_rate, path='./data/gait_outcomes.csv'):
    train_index, val_index, test_index, del_index, labels = get_train_val_tes_index(test_rate, val_rate, path)

    # load data
    df = pd.read_csv(path)
    df = df.drop(del_index, axis=0)

    data = df.to_numpy()
    # delete the id, age, and other duplicate features
    features = np.delete(data, [0, 1, 6, 7, 8, 9, 11, 21, 22, 23, 31, 32, 33, 41, 42, 43, 50, 51, 52], axis=1)

    # form the dataset
    X_train = features[train_index]
    y_train = labels[train_index]

    X_val = features[val_index]
    y_val = labels[val_index]

    X_test = features[test_index]
    y_test = labels[test_index]

    transformer = preprocessing.RobustScaler().fit(X_train)
    X_train = transformer.transform(X_train)
    X_val = transformer.transform(X_val)
    X_test = transformer.transform(X_test)

    # hyperparameters from KPCA_tuning.py
    kpca = KernelPCA(gamma=0.01, kernel='rbf',n_components=17).fit(X_train)

    X_train = kpca.transform(X_train)
    X_val = kpca.transform(X_val)
    X_test = kpca.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


# gets the dataset for deep learning
def get_dl_dataset(test_rate, val_rate, shuffle_flag=1, path='./data/norm_dataset.mat', csv_path='./data/gait_outcomes.csv'):
    train_index, val_index, test_index, del_index, labels = get_train_val_tes_index(test_rate, val_rate, csv_path, shuffle_flag)

    # load data
    segments, _, _ = load_data(path)
    # remove bad data
    segments = np.delete(segments, del_index, axis=0)
    # form the dataset
    X_train = segments[train_index]
    y_train = labels[train_index]

    X_val = segments[val_index]
    y_val = labels[val_index]

    X_test = segments[test_index]
    y_test = labels[test_index]


    return X_train, X_val, X_test, y_train, y_val, y_test


# split or merge the windows length of the segments
# window size: 1024/k (k=0.5 ->2048 samples; k=1 -> 1024; k=2 ->512; k=4 -> 256; k=8 ->128)
def segment_length_split(X_train, X_val, X_test, y_train, y_val, y_test, k, shuffle_flag):
    # merge segments to form a bigger window size
    if k<1:
        if shuffle_flag == 0:
            n_size = int(1/k)
            X_train_new = np.array([np.vstack((X_train[n_size * m: n_size * (m + 1), :, :]))
                                for m in range(int(len(X_train) / n_size))])
            X_val_new = np.array([np.vstack((X_val[n_size * m: n_size * (m + 1), :, :]))
                              for m in range(int(len(X_val) / n_size))])
            X_test_new = np.array([np.vstack((X_test[n_size * m: n_size * (m + 1), :, :]))
                               for m in range(int(len(X_test) / n_size))])
            y_train_new = np.array([y_train[n_size * m] for m in range(int(len(y_train) / n_size))])
            y_val_new = np.array([y_val[n_size * m] for m in range(int(len(y_val) / n_size))])
            y_test_new = np.array([y_test[n_size * m] for m in range(int(len(y_test) / n_size))])
        else:
            print('need no shuffle')
    # split segment to a smaller window size
    if k > 1:
        X_train_new, X_val_new, X_test_new, y_train_new, y_val_new, y_test_new = [], [], [], [], [], []
        X_train = np.split(X_train, k, axis=1)
        X_val = np.split(X_val, k, axis=1)
        X_test = np.split(X_test, k, axis=1)

        for i in range(0, int(k)):
            X_train_new.extend(X_train[i])
            X_val_new.extend(X_val[i])
            X_test_new.extend(X_test[i])
            y_train_new.extend(y_train)
            y_val_new.extend(y_val)
            y_test_new.extend(y_test)

        X_train_new = np.array(X_train_new)
        X_val_new = np.array(X_val_new)
        X_test_new = np.array(X_test_new)
        y_train_new = np.array(y_train_new)
        y_val_new = np.array(y_val_new)
        y_test_new = np.array(y_test_new)

    if k == 1:
        X_train_new = X_train
        X_val_new = X_val
        X_test_new = X_test
        y_train_new = y_train
        y_val_new = y_val
        y_test_new = y_test
    return X_train_new, X_val_new, X_test_new, y_train_new, y_val_new, y_test_new


def median_filter(data, f_size):
    # @author: Kemeng Chen
    # https://github.com/KChen89/Accelerometer-Filtering
    lgth, num_signal = data.shape
    f_data = np.zeros([lgth, num_signal])
    for i in range(num_signal):
        f_data[:, i] = signal.medfilt(data[:, i], f_size)
    return f_data


def freq_filter(data, f_size, cutoff):
    # @author: Kemeng Chen
    # https://github.com/KChen89/Accelerometer-Filtering
    lgth, num_signal = data.shape
    f_data = np.zeros([lgth, num_signal])
    lpf = signal.firwin(f_size, cutoff, window='hamming')
    for i in range(num_signal):
        f_data[:, i] = signal.convolve(data[:, i], lpf, mode='same')
    return f_data


# plots confusion matrix
def plot_cm(cm, label, title):
    fig, ax = plt.subplots()
    im, cbar, cf_percentages, accuracy = heatmap(cm, label, label, title=title,
                                                 ax=ax,
                                                 cmap="Blues", cbarlabel="percentage [%]")
    annotate_heatmap(im, cf_percentages, cm, valfmt="{x:.1f} t")
    fig.tight_layout()


def heatmap(data, row_labels, col_labels, title, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    cf_percentages = data.astype('float') / data.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cf_percentages * 100, **kwargs)
    im.set_clim(0, 100)
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=15, **csfont)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    ax.set_xticklabels(col_labels, fontsize=13, **csfont)
    ax.set_yticklabels(row_labels, fontsize=13, **csfont)

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.ylabel('True label', fontsize=15, **csfont)
    plt.xlabel('Predicted label', fontsize=15, **csfont)
    accuracy = np.trace(data) / float(np.sum(data))
    stats_text = "\nAccuracy={:0.1%}".format(accuracy)
    plt.title(title + stats_text,fontsize=15, **csfont)

    return im, cbar, cf_percentages, accuracy


def annotate_heatmap(im, cf_percentages, data=None, valfmt="{x:.1f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    if threshold is not None:
        threshold = im.norm(cf_percentages)
    else:
        threshold = cf_percentages.max() / 2.

    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Change the text's color depending on the data.
    cf_percentages = data.astype('float') / data.sum(axis=1)[:, np.newaxis]
    group_percentages = ["{0:.1%} \n".format(value) for value in cf_percentages.flatten()]
    group_counts = ["{0:0.0f}\n".format(value) for value in data.flatten()]
    box_labels = [f"{v1}{v2}".strip() for v1, v2 in zip(group_percentages, group_counts)]
    box_labels = np.asarray(box_labels).reshape(data.shape[0], data.shape[1])

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(cf_percentages[i, j] > threshold)])
            text = im.axes.text(j, i, box_labels[i, j], **kw)
            texts.append(text)

    return texts


# plot roc curve for one classification results
def plot_roc(fpr, tpr, title):
    plt.figure(figsize=(7, 6))
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, c='r', lw=3, alpha=0.7, label=u'AUC=%.2f' % auc)
    plt.plot((0, 1), (0, 1), c='b', lw=2, ls='--', alpha=0.7, label='baseline = 0.5')
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.3)
    ax.spines['left'].set_linewidth(1.3)
    ax.spines['right'].set_linewidth(1.3)
    ax.spines['top'].set_linewidth(1.3)
    plt.xlabel('False Positive Rate', fontsize=15, **csfont)
    plt.ylabel('True Positive Rate', fontsize=15, **csfont)
    plt.grid()
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=14)
    plt.title(title, fontsize=16, **csfont)
    return auc


# plot several roc curves
def plot_roc_compare(fpr_list, tpr_list, model_name_list, auc_list, title):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    marker_style = ['o', 'v', 's', 'X', 'd', 'P']
    plt.figure(figsize=(7, 6))
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'

    plt.plot((0, 1), (0, 1), c='b', lw=2, ls='--', alpha=0.7, label='baseline = 0.5')
    for i in range(len(model_name_list)):
        plt.plot(fpr_list[i], tpr_list[i], lw=2, alpha=0.7, c=colors[i],
                 marker=marker_style[i], markevery=7, markersize=9,
                 label=model_name_list[i] + u'=%.2f' % auc_list[i])
        L =plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=18)
        plt.setp(L.texts, **csfont)

    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1),  fontsize=12)
    plt.yticks(np.arange(0, 1.1, 0.1),  fontsize=12)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.3)
    ax.spines['left'].set_linewidth(1.3)
    ax.spines['right'].set_linewidth(1.3)
    ax.spines['top'].set_linewidth(1.3)
    plt.xlabel('False Positive Rate', fontsize=22, **csfont)
    plt.ylabel('True Positive Rate', fontsize=22, **csfont)
    plt.grid()
    plt.title(title, fontsize=22, **csfont)
    plt.tight_layout()

# get the walking frequency from the 1024 size segements
df = 100 / (1024 - 1)
def get_fre_list(data):
    fre_list = []
    for i in range(0, len(data)):
        power = np.fft.fft(data[i,:,2])
        power = abs(power[:int(power.size / 2)])
        power[0] = 0
        fre = df * np.argmax(power)
        fre_list.append(fre)
    fre_list = np.array(fre_list)
    mean_fre_list = [np.mean(fre_list[m*10:(m+1)*10]) for m in range(0, int(len(fre_list)/10))]
    return mean_fre_list

# normalize the gait frequency
def resample(fre_list, data, k):
    seg_number = int(10*k)
    for i in range(0, len(fre_list)):
        # the number of data points in one step, define by the walking frequenchy
        one_step_sample_num = math.ceil(1/fre_list[i]*100)
        # the step number of each window size: 1.2 step for 128, 2.4 for 256, 4.8 for 512
        normalized_length = int(8/k * 1.2 * one_step_sample_num)
        for j in range(i*seg_number, (i+1)*seg_number):
            x = data[j, 0:normalized_length, 0]
            y = data[j, 0:normalized_length, 1]
            z = data[j, 0:normalized_length, 2]

            f_x = interpolate.UnivariateSpline(range(0,normalized_length),x,s=0)
            f_y = interpolate.UnivariateSpline(range(0,normalized_length),y,s=0)
            f_z = interpolate.UnivariateSpline(range(0,normalized_length),z,s=0)

            data[j, :, 0] = f_x(np.linspace(0, normalized_length, int(1024/k)))
            data[j, :, 1] = f_y(np.linspace(0, normalized_length, int(1024/k)))
            data[j, :, 2] = f_z(np.linspace(0, normalized_length, int(1024/k)))

    return data