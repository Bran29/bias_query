from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from generateSynthet import generate_dataset_1, generate_dataset_2
from Adult import generate_Adult_sample,add_noise
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import random
from COMPAS import generate_Compas_sample
import scipy.stats as stats
from xgboost import XGBClassifier
import math


def counterfacts_dataset(D_p):
    D_n = D_p.copy()
    D_n[:, -2] = 1 - D_n[:, -2]
    return D_n


def Adult_UEffect(K_max):

    Ua_all = []
    Acc_all = []
    
    (
        D_init_x,
        D_init_y,
        _,
        _,
        D_test_x,
        D_test_y,
    ) = generate_Adult_sample(D_init_num=1, D_providers_num=5)
    D_init_y[0]=add_noise(D_init_y[0],0.5)
    predictor = LogisticRegression(random_state=0)
    for k in range(K_max):
        (
        _,
        _,
        D_provider_x,
        D_provider_y,
        _,
        _,
    ) = generate_Adult_sample(D_init_num=1, D_providers_num=5)
        for ii in range(len(D_provider_y)):
            D_provider_y[ii]=add_noise(D_provider_y[ii],random.random())
        
        Ua_i = []
        Acc_i = []
        for ii in range(5):
            predictor.fit(D_init_x[0], D_init_y[0])
            y_pred = predictor.predict(D_provider_x[ii])
            Ua = accuracy_score(D_provider_y[ii], y_pred)

            predictor.fit(
                np.concatenate((D_init_x[0], D_provider_x[ii])),
                np.concatenate((D_init_y[0], D_provider_y[ii])),
            )
            y_pred = predictor.predict(D_test_x[0])
            Acc = accuracy_score(D_test_y[0], y_pred)
            Ua_i.append(Ua)
            Acc_i.append(Acc)
        Ua_all.append(Ua_i)
        Acc_all.append(Acc_i)
    Ua_all = np.array(Ua_all)
    Acc_all = np.array(Acc_all)

    Ua_all_mean = np.mean(Ua_all, axis=0)
    Acc_all_mean = np.mean(Acc_all, axis=0)
    Ua_all_std = np.std(Ua_all, axis=0)
    Acc_all_std = np.std(Acc_all, axis=0)

    pearson_corr, p_value = stats.pearsonr(Ua_all_mean, Acc_all_mean)
    print("pearson_corr: ", pearson_corr)
    print("p_value: ", p_value)
    fig, axs = plt.subplots(1, 1, figsize=(3, 3), sharey=True)
    axs.plot(list(range(5)), Ua_all_mean, label="Uf")
    axs.plot(list(range(5)), Acc_all_mean, label="DP")
    axs.fill_between(list(range(5)), Ua_all_mean - Ua_all_std, Ua_all_mean + Ua_all_std)
    axs.fill_between(list(range(5)), Acc_all_mean - Acc_all_std, Acc_all_mean + Acc_all_std)
    plt.legend()
    plt.savefig("fig/Adult_ua_effect.jpg")


if __name__ == "__main__":
    Adult_UEffect(1)
