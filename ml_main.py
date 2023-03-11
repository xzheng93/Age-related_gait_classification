import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import random
from Utils import get_ml_dataset, RANDOM_STATE, plot_cm, plot_roc, plot_roc_compare
import os
import pickle

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# evaluate all the fine-tuning conventional machine learning models
# get all accuracy metric and plot roc and confusion matrix figures
if __name__ == '__main__':
    # load dataset
    X_train, X_val, X_test, y_train, y_val, y_test = get_ml_dataset(0.2, 0.1, path='./data/gait_outcomes.csv')

    y_train = np.argmax(y_train, axis=1)
    y_val = np.argmax(y_val, axis=1)
    y_test = np.argmax(y_test, axis=1)

    plot_save_path = f'./result/model_plots'

    ml_mode =['nb', 'rf', 'knn', 'svc']

    fpr_list, tpr_list, auc_list, model_name_list = [], [], [], []
    for model_name in ml_mode:
        print(f'======================={model_name}====================')
        if os.path.exists(f'./result/models/ml/{model_name}.pickle'):
            with open(f'./result/models/ml/{model_name}.pickle', 'rb') as file:
                model = pickle.load(file)
            print(f'loading model {model_name}')
        else:
            break

        # evaluation
        y_pre = model.predict(X_test)
        acc = metrics.accuracy_score(y_test, y_pre)
        pre = metrics.precision_score(y_test, y_pre, average="weighted")
        f1 = metrics.f1_score(y_test, y_pre, average="weighted")
        rec = metrics.recall_score(y_test, y_pre, average="weighted")
        print(f'Precision: {pre}')
        print(f'Recall: {rec}')
        print(f'F1_score: {f1}')
        print(f'Testing Accuracy: {acc}')

        # plot cm
        cm = metrics.confusion_matrix(y_test, y_pre)
        label = ['Adult', 'Older adult']
        plot_cm(cm, label, '') #f'Confusion Matrix {model_name}'
        plt.savefig(f'{plot_save_path}/Confusion Matrix {model_name}.pdf', format='pdf')
        plt.show()

        # plot roc
        pred_probability= model.predict_proba(X_test)
        pred_probability_ = pred_probability[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_probability_)

        auc = plot_roc(fpr, tpr, f'The ROC and AUC for {model_name}')
        plt.savefig(f'{plot_save_path}/ROC_{model_name}.pdf', format='pdf')
        plt.show()

        print(f'AUC: {auc}')
        auc_list.append(auc)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        model_name_list.append(model_name)
    # compare roc of different models in one fig
    plot_roc_compare(fpr_list, tpr_list, model_name_list, auc_list, 'ROC comparison of machine learning')
    plt.savefig(f'{plot_save_path}/ROC comparison of machine learning.pdf', format='pdf')
    plt.show()


