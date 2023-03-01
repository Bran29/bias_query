from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from generateSynthet import generate_dataset_1, generate_dataset_2
from Adult import generate_Adult_sample
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import random
from COMPAS import generate_Compas_sample
import scipy.stats as stats
from xgboost import XGBClassifier
import math
from tqdm import tqdm


def counterfacts_dataset(D_p):
    D_n = D_p.copy()
    D_n[:, -2] = 1 - D_n[:, -2]
    return D_n


def Adult_UEffect(K_max):

    Uf_all = []
    DP_all = []
    for k in tqdm(range(K_max)):

        (
            D_init_x,
            D_init_y,
            D_provider_x,
            D_provider_y,
            D_test_x,
            D_test_y,
        ) = generate_Compas_sample(D_init_num=1, D_providers_num=5)
        predictor = MLPClassifier(hidden_layer_sizes=5, max_iter=20, random_state=k)
        Uf_i = []
        DP_i = []
        for ii in tqdm(range(5)):
            predictor.fit(D_init_x[0], D_init_y[0])

            y_pred = predictor.predict(D_provider_x[ii])
            accuracy = accuracy_score(D_provider_y[ii], y_pred)

            y_pred_z1 = y_pred[np.where(D_provider_x[ii][:, 4] > 0)[0]]
            y_pred_z0 = y_pred[np.where(D_provider_x[ii][:, 4] <= 0)[0]]
            Uf = abs(
                np.count_nonzero(y_pred_z1 == 1)
                / np.count_nonzero(D_provider_x[ii][:, 4] > 0)
                - np.count_nonzero(y_pred_z0 == 1)
                / np.count_nonzero(D_provider_x[ii][:, 4] <= 0)
            )

            predictor.fit(
                np.concatenate((D_init_x[0], D_provider_x[ii])),
                np.concatenate((D_init_y[0], D_provider_y[ii])),
            )
            y_pred = predictor.predict(D_test_x[0])
            accuracy = accuracy_score(D_test_y[0], y_pred)

            y_pred_z1 = y_pred[np.where(D_test_x[0][:, 4] > 0)[0]]
            y_pred_z0 = y_pred[np.where(D_test_x[0][:, 4] <= 0)[0]]
            DP = abs(
                np.count_nonzero(y_pred_z1 == 1)
                / np.count_nonzero(D_test_x[0][:, 4] > 0)
                - np.count_nonzero(y_pred_z0 == 1)
                / np.count_nonzero(D_test_x[0][:, 4] <= 0)
            )
            Uf_i.append(Uf)
            DP_i.append(DP)
        Uf_all.append(Uf_i)
        DP_all.append(DP_i)
    Uf_all = np.array(Uf_all)
    DP_all = np.array(DP_all)

    Uf_all_mean = np.mean(Uf_all, axis=0)
    DP_all_mean = np.mean(DP_all, axis=0)
    Uf_all_std = np.std(Uf_all, axis=0)
    DP_all_std = np.std(DP_all, axis=0)

    pearson_corr, p_value = stats.pearsonr(Uf_all_mean, DP_all_mean)
    print("pearson_corr: ", pearson_corr)
    print("p_value: ", p_value)
    fig, axs = plt.subplots(1, 1, figsize=(3, 3), sharey=True)
    axs.plot(list(range(5)), Uf_all_mean, label="Uf")
    axs.plot(list(range(5)), DP_all_mean, label="DP")
    axs.fill_between(list(range(5)), Uf_all_mean - Uf_all_std, Uf_all_mean + Uf_all_std)
    axs.fill_between(list(range(5)), DP_all_mean - DP_all_std, DP_all_mean + DP_all_std)
    plt.legend()
    plt.savefig("fig/compass_uf_effect.jpg")


if __name__ == "__main__":
    Adult_UEffect(20)
