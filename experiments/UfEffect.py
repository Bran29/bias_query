# import
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from generateSynthet import generate_dataset_1, generate_dataset_2
from Adult import generate_Adult_sample
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import random
from COMPAS import generate_Compas_sample


def UfEffect(K_max):
    dp_upd1 = []
    dp_upd2 = []
    dp_new1 = []
    dp_new2 = []

    acc_upd1 = []
    acc_upd2 = []
    acc_new1 = []
    acc_new2 = []

    clf = SVC(kernel="linear", random_state=0)

    Synset_test = generate_dataset_1(False, 100, np.pi / 2)
    bias_level = 0.5
    Synset_init = generate_dataset_1(False, 100, np.pi / 4)
    for kk in range(K_max):
        bias_level = 0
        Synset_new = np.concatenate(
            (
                generate_dataset_1(False, int(bias_level * 1000), 0),
                generate_dataset_2(
                    False,
                    int((1 - bias_level) * 1000),
                    0,
                ),
            ),
            axis=0,
        )
        bias_level = 1
        Synset_new2 = np.concatenate(
            (
                generate_dataset_1(False, int(bias_level * 1000), 0),
                generate_dataset_2(False, int((1 - bias_level) * 1000), 0),
            ),
            axis=0,
        )  # bias to z=0
        # create and train SVM

        # pre-test
        clf.fit(Synset_init[:, :3], Synset_init[:, 3])

        y_pred = clf.predict(Synset_new[:, :3])
        accuracy = accuracy_score(Synset_new[:, 3], y_pred)

        y_pred_z1 = y_pred[np.where(Synset_new[:, 2] == 1)[0]]
        y_pred_z0 = y_pred[np.where(Synset_new[:, 2] == 0)[0]]
        dp_new1.append(
            np.count_nonzero(y_pred_z1 == 1) / np.count_nonzero(Synset_new[:, 2] == 1)
            - np.count_nonzero(y_pred_z0 == 1) / np.count_nonzero(Synset_new[:, 2] == 0)
        )
        acc_new1.append(accuracy)
        # on new2 test
        y_pred = clf.predict(Synset_new2[:, :3])
        accuracy = accuracy_score(Synset_new2[:, 3], y_pred)

        y_pred_z1 = y_pred[np.where(Synset_new2[:, 2] == 1)[0]]
        y_pred_z0 = y_pred[np.where(Synset_new2[:, 2] == 0)[0]]
        dp_new2.append(
            np.count_nonzero(y_pred_z1 == 1) / np.count_nonzero(Synset_new2[:, 2] == 1)
            - np.count_nonzero(y_pred_z0 == 1)
            / np.count_nonzero(Synset_new2[:, 2] == 0)
        )
        acc_new2.append(accuracy)
        # new1-trained
        Synset_init_new = np.concatenate((Synset_init, Synset_new), axis=0)
        clf.fit(Synset_init_new[:, :3], Synset_init_new[:, 3])
        y_pred = clf.predict(Synset_test[:, :3])
        accuracy = accuracy_score(Synset_test[:, 3], y_pred)

        y_pred_z1 = y_pred[np.where(Synset_test[:, 2] == 1)[0]]
        y_pred_z0 = y_pred[np.where(Synset_test[:, 2] == 0)[0]]
        dp_upd1.append(
            np.count_nonzero(y_pred_z1 == 1) / np.count_nonzero(Synset_test[:, 2] == 1)
            - np.count_nonzero(y_pred_z0 == 1)
            / np.count_nonzero(Synset_test[:, 2] == 0)
        )
        acc_upd1.append(accuracy)
        # new2-trained
        Synset_init_new = np.concatenate((Synset_init, Synset_new2), axis=0)
        clf.fit(Synset_init_new[:, :3], Synset_init_new[:, 3])
        y_pred = clf.predict(Synset_test[:, :3])
        accuracy = accuracy_score(Synset_test[:, 3], y_pred)

        y_pred_z1 = y_pred[np.where(Synset_test[:, 2] == 1)[0]]
        y_pred_z0 = y_pred[np.where(Synset_test[:, 2] == 0)[0]]
        dp_upd2.append(
            np.count_nonzero(y_pred_z1 == 1) / np.count_nonzero(Synset_test[:, 2] == 1)
            - np.count_nonzero(y_pred_z0 == 1)
            / np.count_nonzero(Synset_test[:, 2] == 0)
        )
        acc_upd2.append(accuracy)
        print(kk, "iteration end")

    return (
        dp_new1,
        dp_upd1,
        dp_new2,
        dp_upd2,
    )


def UfEffect(K_max):
    dp_upd1 = []
    dp_upd2 = []
    dp_new1 = []
    dp_new2 = []

    acc_upd1 = []
    acc_upd2 = []
    acc_new1 = []
    acc_new2 = []

    clf = SVC(kernel="linear", random_state=0)
    Synset_test = np.concatenate(
        (
            generate_dataset_1(False, 100, np.pi / 4),
            generate_dataset_2(False, 100, np.pi / 4),
        ),
        axis=0,
    )  # fair
    Synset_test = generate_dataset_1(False, 100, np.pi / 2)
    bias_level = 0.5
    Synset_init = np.concatenate(
        (
            generate_dataset_1(False, int(bias_level * 100), np.pi / 4),
            generate_dataset_2(False, int((1 - bias_level) * 100), np.pi / 4),
        ),
        axis=0,
    )
    for kk in range(K_max):
        bias_level = 0
        Synset_new = np.concatenate(
            (
                generate_dataset_1(False, int(bias_level * 1000), 0),
                generate_dataset_2(
                    False,
                    int((1 - bias_level) * 1000),
                    0,
                ),
            ),
            axis=0,
        )
        bias_level = 1
        Synset_new2 = np.concatenate(
            (
                generate_dataset_1(False, int(bias_level * 1000), 0),
                generate_dataset_2(False, int((1 - bias_level) * 1000), 0),
            ),
            axis=0,
        )  # bias to z=0
        # create and train SVM

        # pre-test
        clf.fit(Synset_init[:, :3], Synset_init[:, 3])

        y_pred = clf.predict(Synset_new[:, :3])
        accuracy = accuracy_score(Synset_new[:, 3], y_pred)

        y_pred_z1 = y_pred[np.where(Synset_new[:, 2] == 1)[0]]
        y_pred_z0 = y_pred[np.where(Synset_new[:, 2] == 0)[0]]
        dp_new1.append(
            np.count_nonzero(y_pred_z1 == 1) / np.count_nonzero(Synset_new[:, 2] == 1)
            - np.count_nonzero(y_pred_z0 == 1) / np.count_nonzero(Synset_new[:, 2] == 0)
        )
        acc_new1.append(accuracy)
        # on new2 test
        y_pred = clf.predict(Synset_new2[:, :3])
        accuracy = accuracy_score(Synset_new2[:, 3], y_pred)

        y_pred_z1 = y_pred[np.where(Synset_new2[:, 2] == 1)[0]]
        y_pred_z0 = y_pred[np.where(Synset_new2[:, 2] == 0)[0]]
        dp_new2.append(
            np.count_nonzero(y_pred_z1 == 1) / np.count_nonzero(Synset_new2[:, 2] == 1)
            - np.count_nonzero(y_pred_z0 == 1)
            / np.count_nonzero(Synset_new2[:, 2] == 0)
        )
        acc_new2.append(accuracy)
        # new1-trained
        Synset_init_new = np.concatenate((Synset_init, Synset_new), axis=0)
        clf.fit(Synset_init_new[:, :3], Synset_init_new[:, 3])
        y_pred = clf.predict(Synset_test[:, :3])
        accuracy = accuracy_score(Synset_test[:, 3], y_pred)

        y_pred_z1 = y_pred[np.where(Synset_test[:, 2] == 1)[0]]
        y_pred_z0 = y_pred[np.where(Synset_test[:, 2] == 0)[0]]
        dp_upd1.append(
            np.count_nonzero(y_pred_z1 == 1) / np.count_nonzero(Synset_test[:, 2] == 1)
            - np.count_nonzero(y_pred_z0 == 1)
            / np.count_nonzero(Synset_test[:, 2] == 0)
        )
        acc_upd1.append(accuracy)
        # new2-trained
        Synset_init_new = np.concatenate((Synset_init, Synset_new2), axis=0)
        clf.fit(Synset_init_new[:, :3], Synset_init_new[:, 3])
        y_pred = clf.predict(Synset_test[:, :3])
        accuracy = accuracy_score(Synset_test[:, 3], y_pred)

        y_pred_z1 = y_pred[np.where(Synset_test[:, 2] == 1)[0]]
        y_pred_z0 = y_pred[np.where(Synset_test[:, 2] == 0)[0]]
        dp_upd2.append(
            np.count_nonzero(y_pred_z1 == 1) / np.count_nonzero(Synset_test[:, 2] == 1)
            - np.count_nonzero(y_pred_z0 == 1)
            / np.count_nonzero(Synset_test[:, 2] == 0)
        )
        acc_upd2.append(accuracy)
        print(kk, "iteration end")

    return (
        dp_new1,
        dp_upd1,
        dp_new2,
        dp_upd2,
    )


def UfEffect_Adult(K_max):
    dp_upd1 = []
    dp_upd2 = []
    dp_new1 = []
    dp_new2 = []

    acc_upd1 = []
    acc_upd2 = []
    acc_new1 = []
    acc_new2 = []

    predictor = MLPClassifier(max_iter=100, random_state=0)
    # predictor = SVC(kernel="linear", random_state=0)
    # load dataset
    (
        D_init_x,
        D_init_y,
        _,
        _,
        D_test_x,
        D_test_y,
    ) = generate_Adult_sample(D_init_num=1, D_providers_num=2)

    for kk in range(K_max):

        # load dataset
        (
            _,
            _,
            D_providers_x,
            D_providers_y,
            _,
            _,
        ) = generate_Adult_sample(D_init_num=1, D_providers_num=2)

        # create and train MLP

        # pre-test
        predictor.fit(D_init_x[0], D_init_y[0])

        # on new1 test
        y_pred = predictor.predict(D_providers_x[0])
        accuracy = accuracy_score(D_providers_y[0], y_pred)

        y_pred_z1 = y_pred[np.where(D_providers_x[0][:, -5] > 0)[0]]
        y_pred_z0 = y_pred[np.where(D_providers_x[0][:, -5] < 0)[0]]
        dp_new1.append(
            np.count_nonzero(y_pred_z1 == 1)
            / np.count_nonzero(D_providers_x[0][:, -5] > 0)
            - np.count_nonzero(y_pred_z0 == 1)
            / np.count_nonzero(D_providers_x[0][:, -5] < 0)
        )
        acc_new1.append(accuracy)
        # on new2 test
        y_pred = predictor.predict(D_providers_x[1])
        accuracy = accuracy_score(D_providers_y[1], y_pred)

        y_pred_z1 = y_pred[np.where(D_providers_x[1][:, -5] > 0)[0]]
        y_pred_z0 = y_pred[np.where(D_providers_x[1][:, -5] < 0)[0]]
        dp_new2.append(
            np.count_nonzero(y_pred_z1 == 1)
            / np.count_nonzero(D_providers_x[1][:, -5] > 0)
            - np.count_nonzero(y_pred_z0 == 1)
            / np.count_nonzero(D_providers_x[1][:, -5] < 0)
        )
        acc_new2.append(accuracy)
        # new1-trained
        Adult_init_new_x = np.concatenate((D_init_x[0], D_providers_x[0]), axis=0)
        Adult_init_new_y = np.concatenate((D_init_y[0], D_providers_y[0]), axis=0)
        predictor.fit(Adult_init_new_x, Adult_init_new_y)
        y_pred = predictor.predict(D_test_x[0])
        accuracy = accuracy_score(D_test_y[0], y_pred)

        y_pred_z1 = y_pred[np.where(D_test_x[0][:, -5] > 0)[0]]
        y_pred_z0 = y_pred[np.where(D_test_x[0][:, -5] < 0)[0]]
        dp_upd1.append(
            np.count_nonzero(y_pred_z1 == 1) / np.count_nonzero(D_test_x[0][:, -5] > 0)
            - np.count_nonzero(y_pred_z0 == 1)
            / np.count_nonzero(D_test_x[0][:, -5] < 0)
        )
        acc_upd1.append(accuracy)
        # new2-trained
        Adult_init_new_x = np.concatenate((D_init_x[0], D_providers_x[1]), axis=0)
        Adult_init_new_y = np.concatenate((D_init_y[0], D_providers_y[1]), axis=0)
        predictor.fit(Adult_init_new_x, Adult_init_new_y)
        y_pred = predictor.predict(D_test_x[0])
        accuracy = accuracy_score(D_test_y[0], y_pred)

        y_pred_z1 = y_pred[np.where(D_test_x[0][:, -5] > 0)[0]]
        y_pred_z0 = y_pred[np.where(D_test_x[0][:, -5] < 0)[0]]
        dp_upd2.append(
            np.count_nonzero(y_pred_z1 == 1) / np.count_nonzero(D_test_x[0][:, -5] > 0)
            - np.count_nonzero(y_pred_z0 == 1)
            / np.count_nonzero(D_test_x[0][:, -5] < 0)
        )
        acc_upd2.append(accuracy)
        print(kk, "iteration end")

    return (
        dp_new1,
        dp_upd1,
        dp_new2,
        dp_upd2,
        acc_new1,
        acc_upd1,
        acc_new2,
        acc_upd2,
    )


def UfEffect_Compas(K_max):
    dp_upd1 = []
    dp_upd2 = []
    dp_new1 = []
    dp_new2 = []

    acc_upd1 = []
    acc_upd2 = []
    acc_new1 = []
    acc_new2 = []

    predictor = MLPClassifier(max_iter=100, random_state=0)

    # load dataset
    (
        D_init_x,
        D_init_y,
        _,
        _,
        D_test_x,
        D_test_y,
    ) = generate_Compas_sample(D_init_num=1, D_providers_num=2)

    for kk in range(K_max):

        # load dataset
        (
            _,
            _,
            D_providers_x,
            D_providers_y,
            _,
            _,
        ) = generate_Compas_sample(D_init_num=1, D_providers_num=2)

        # create and train MLP

        # pre-test
        predictor.fit(D_init_x[0], D_init_y[0])

        # on new1 test
        y_pred = predictor.predict(D_providers_x[0])
        accuracy = accuracy_score(D_providers_y[0], y_pred)

        y_pred_z1 = y_pred[np.where(D_providers_x[0][:, 2] > 0)[0]]
        y_pred_z0 = y_pred[np.where(D_providers_x[0][:, 2] < 0)[0]]
        dp_new1.append(
            np.count_nonzero(y_pred_z1 == 1)
            / np.count_nonzero(D_providers_x[0][:, 2] > 0)
            - np.count_nonzero(y_pred_z0 == 1)
            / np.count_nonzero(D_providers_x[0][:, 2] < 0)
        )
        acc_new1.append(accuracy)
        # on new2 test
        y_pred = predictor.predict(D_providers_x[1])
        accuracy = accuracy_score(D_providers_y[1], y_pred)

        y_pred_z1 = y_pred[np.where(D_providers_x[1][:, 2] > 0)[0]]
        y_pred_z0 = y_pred[np.where(D_providers_x[1][:, 2] < 0)[0]]
        dp_new2.append(
            np.count_nonzero(y_pred_z1 == 1)
            / np.count_nonzero(D_providers_x[1][:, 2] > 0)
            - np.count_nonzero(y_pred_z0 == 1)
            / np.count_nonzero(D_providers_x[1][:, 2] < 0)
        )
        acc_new2.append(accuracy)
        # new1-trained
        Adult_init_new_x = np.concatenate((D_init_x[0], D_providers_x[0]), axis=0)
        Adult_init_new_y = np.concatenate((D_init_y[0], D_providers_y[0]), axis=0)
        predictor.fit(Adult_init_new_x, Adult_init_new_y)
        y_pred = predictor.predict(D_test_x[0])
        accuracy = accuracy_score(D_test_y[0], y_pred)

        y_pred_z1 = y_pred[np.where(D_test_x[0][:, 2] > 0)[0]]
        y_pred_z0 = y_pred[np.where(D_test_x[0][:, 2] < 0)[0]]
        dp_upd1.append(
            np.count_nonzero(y_pred_z1 == 1) / np.count_nonzero(D_test_x[0][:, 2] > 0)
            - np.count_nonzero(y_pred_z0 == 1) / np.count_nonzero(D_test_x[0][:, 2] < 0)
        )
        acc_upd1.append(accuracy)
        # new2-trained
        Adult_init_new_x = np.concatenate((D_init_x[0], D_providers_x[1]), axis=0)
        Adult_init_new_y = np.concatenate((D_init_y[0], D_providers_y[1]), axis=0)
        predictor.fit(Adult_init_new_x, Adult_init_new_y)
        y_pred = predictor.predict(D_test_x[0])
        accuracy = accuracy_score(D_test_y[0], y_pred)

        y_pred_z1 = y_pred[np.where(D_test_x[0][:, 2] > 0)[0]]
        y_pred_z0 = y_pred[np.where(D_test_x[0][:, 2] < 0)[0]]
        dp_upd2.append(
            np.count_nonzero(y_pred_z1 == 1) / np.count_nonzero(D_test_x[0][:, 2] > 0)
            - np.count_nonzero(y_pred_z0 == 1) / np.count_nonzero(D_test_x[0][:, 2] < 0)
        )
        acc_upd2.append(accuracy)
        print(kk, "iteration end")

    return (
        dp_new1,
        dp_upd1,
        dp_new2,
        dp_upd2,
        acc_new1,
        acc_upd1,
        acc_new2,
        acc_upd2,
    )


if __name__ == "__main__":
    (dp_new1_p, dp_upd1_p, dp_new2_p, dp_upd2_p) = UfEffect_Adult(10)
    print("end")
