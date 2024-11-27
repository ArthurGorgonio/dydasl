import statistics
from time import time

import numpy as np
import scipy as sp
from pandas import read_csv
from sklearn.metrics import (
    accuracy_score,
)
from sklearn.metrics import cohen_kappa_score as kappa
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
)
from skmultiflow.data import DataStream
from skmultiflow.trees import HoeffdingTreeClassifier

from src.utils import Log


def Unlabeling_data(X_train, Y_train, Percentage, chunk_size, class_count):
    labeled_count = round(Percentage * chunk_size)
    TLabeled = X_train[0: labeled_count - 1]
    Y_TLabeled = Y_train[0: labeled_count - 1]
    X_Unlabeled = X_train[labeled_count: Y_train.shape[0] - 1]

    cal_count = round(0.3 * TLabeled.shape[0])
    X_cal = TLabeled[0: cal_count - 1]
    Y_cal = Y_TLabeled[0: cal_count - 1]
    X_L = TLabeled[cal_count: TLabeled.shape[0] - 1]
    Y_L = Y_TLabeled[cal_count: TLabeled.shape[0] - 1]

    return X_Unlabeled, X_L, Y_L, X_cal, Y_cal


def Prediction_by_CP(classifier, X, Y, X_Unlabeled, class_count, sl):
    row = X_Unlabeled.shape[0]
    col = class_count
    p_values = np.zeros([row, col])
    labels = np.ones((row, col), dtype=bool)
    alphas = NCM(classifier, X, Y, 1, class_count)

    for elem in range(row):
        c = []

        for o in class_set:
            a_test = NCM(
                classifier,
                np.array([X_Unlabeled[elem, :]]),
                o,
                2,
                class_count,
            )
            idx = np.argwhere(Y == o).flatten()
            temp = alphas[idx]
            p = len(temp[temp >= a_test])

            if idx.shape[0] == 0:
                s = 0
            else:
                s = p / idx.shape[0]
                print(f's: {s}')
            c.append(s)

            if s < sl:
                labels[elem, int(o) - 1] = False
        p_values[elem, :] = np.array(c)

    return p_values, labels


def NCM(classifier, X, Y, t, class_count):
    if t == 1:
        p = np.zeros([X.shape[0], 1])
        alpha = np.zeros([X.shape[0], 1])

        for g in range(X.shape[0]):
            dic_vote = classifier.get_votes_for_instance(X[g, :])
            vote = np.fromiter(dic_vote.values(), dtype=float)
            vote_keys = np.fromiter(dic_vote.keys(), dtype=int)
            Sum = np.sum(vote)
            keys = np.argwhere(vote_keys == int(Y[g])).flatten()

            if keys.size == 0:
                p[g] = (1) / (Sum + class_count)
            else:
                for key, val in dic_vote.items():
                    if key == float(Y[g]):
                        p[g] = (val + 1) / (Sum + class_count)
            alpha[g] = 1 - p[g]

    else:

        dic_vote = classifier.get_votes_for_instance(X[0, :])
        vote = np.fromiter(dic_vote.values(), dtype=float)
        vote_keys = np.fromiter(dic_vote.keys(), dtype=int)
        Sum = np.sum(vote)
        keys = np.argwhere(vote_keys == int(Y)).flatten()

        if keys.size == 0:
            p = (1) / (Sum + class_count)
        else:
            for key, val in dic_vote.items():
                if key == float(Y):
                    p = (val + 1) / (Sum + class_count)
        alpha = 1 - p

    return alpha


def Informatives_selection(X_Unlabeled, p_values, labels, class_count):
    row = X_Unlabeled.shape[0]
    X = np.empty([1, X_Unlabeled.shape[1]])
    Y = np.empty([1])

    for elem in range(row):
        l = np.argwhere(labels[elem, :] == True).flatten()

        if len(l) == 1:
            pp = p_values[elem, l]
            X = np.append(X, [X_Unlabeled[elem, :]], axis=0)
            Y = np.append(Y, [l[0]], axis=0)
    Informatives = X[1: X.shape[0], :]
    Y_Informatives = Y[1: Y.shape[0]]

    return Informatives, Y_Informatives


def Appending_informative_to_nextchunk(
    X_Currentchunk_Labeled,
    Y_Currentchunk_Labeled,
    Informatives,
    Y_Informatives,
):
    X = np.append(X_Currentchunk_Labeled, Informatives, axis=0)
    Y = np.append(Y_Currentchunk_Labeled, Y_Informatives, axis=0)

    return X, Y


def _evaluate_metrics(y_true, y_pred):
    """
    Computa cada uma das métricas adicionadas a partir do valor
    predito pelo comitê.

    Parameters
    ----------
    y_true : ndarray
        Rótulos verdadeiros.
    y_pred : ndarray
        Rótulos preditos pelo comitê.
    """

    for func_name, metric in metrics_calls.items():

        if func_name == "f1":
            metrics[func_name].append(
                metric(y_true, y_pred, average="macro")
            )
        else:
            metrics[func_name].append(
                metric(y_true, y_pred)
            )



def _log_iteration_info(hits, processed, elapsed_time, drift):
    # version = self.detector.__class__
    iteration_info = {
        "ensemble_size": 1,
        "ensemble_hits": hits,
        "drift_detected": drift,
        "instances": processed,
        "elapsed_time": elapsed_time,
        "metrics": {
            "acc": metrics["acc"][-1],
            "f1": metrics["f1"][-1],
            "kappa": metrics["kappa"][-1],
        },
    }
    Log().write_archive_output(**iteration_info)

################################ Main

if __name__ == '__main__':
    datasets = [
        'bank-full_converted.csv',
        'iot-network_converted.csv'
    ]
    chunk_size = 500
    ini_lab = [0.1, 0.5]
    significance_level = 0.98
    metrics = {
        'acc': [],
        'f1': [],
        'kappa': [],
    }
    metrics_calls = {
        'acc': accuracy_score,
        'f1': f1_score,
        'kappa': kappa,
    }

    for Percentage in ini_lab:
        for dataset in datasets:
            dataframe = read_csv(f"datasets/{dataset}")
            dim = dataframe.shape
            array = dataframe.values
            Y = array[1 : dim[0] - 1, dim[1] - 1]
            X = array[1 : dim[0] - 1, 0 : dim[1] - 1]
            class_set = np.unique(Y)
            class_count = np.unique(Y).shape[0]  # number of classess

            dt_name = dataset.split(".", maxsplit=1)[0].split("/")[-1]
            Log().filename = {
                "data_name": dt_name,
                "method_name": f"CPSSDS-{Percentage}"
            }

            Log().write_archive_header()

            stream = DataStream(
                X,
                Y,
                n_targets=class_count,
                cat_features=None,
                name=None,
                allow_nan=False,
            )

            start = time()

            X_chunk1, Y_chunk1 = stream.next_sample(chunk_size)
            t = round(0.2 * X_chunk1.shape[0])
            X_test = X_chunk1[0: t - 1]
            Y_test = Y_chunk1[0: t - 1]
            X_train = X_chunk1[t: X_chunk1.shape[0] - 1]
            Y_train = Y_chunk1[t: X_chunk1.shape[0] - 1]
            num_samples = X.shape[0] - X_chunk1.shape[0]
            [X_U1, X_L1, Y_L1, X_cal1, Y_cal1] = Unlabeling_data(
                X_train, Y_train, Percentage, chunk_size, class_count
            )

            classifier = HoeffdingTreeClassifier()  # num=1
            classifier.fit(X_L1, Y_L1, np.unique(Y))
            sl = 1

            elapsed_time = time() - start

            Y_pred = classifier.predict(X_test)

            _evaluate_metrics(Y_test, Y_pred)
            hits = confusion_matrix(Y_test, Y_pred)
            _log_iteration_info(
                sum(hits.diagonal()),
                stream.sample_idx,
                elapsed_time,
                False
            )


            Kolmogrov = []
            n_samples = 0
            i = 1

            while n_samples < num_samples and stream.has_more_samples():
                start = time()
                p_values, labels = Prediction_by_CP(
                    classifier, X_cal1, Y_cal1, X_U1, class_count, sl
                )
                Informatives, Y_Informatives = Informatives_selection(
                    X_U1, p_values, labels, class_count
                )
                X_Currentchunk, Y_Currentchunk = stream.next_sample(chunk_size)
                t = round(0.2 * X_Currentchunk.shape[0])
                X_test = X_Currentchunk[0: t - 1]
                Y_test = Y_Currentchunk[0: t - 1]
                X_train = X_Currentchunk[t: X_Currentchunk.shape[0] - 1]
                Y_train = Y_Currentchunk[t: X_Currentchunk.shape[0] - 1]
                [
                    X_U2,
                    X_L2,
                    Y_L2,
                    X_cal2,
                    Y_cal2
                ] = Unlabeling_data(
                    X_train, Y_train, Percentage, chunk_size, class_count
                )
                p_values1, labels1 = Prediction_by_CP(
                    classifier, X_cal1, Y_cal1, X_U2, class_count, sl
                )

                if X_Currentchunk.shape[0] >= chunk_size:
                    kst = []
                    class_set = np.unique(Y)

                    for h in class_set:
                        val = sp.stats.ks_2samp(
                            p_values[:, int(h)-1],
                            p_values1[:, int(h)-1]
                        )
                        kst.append(val[1])
                    mean_kst = statistics.mean(kst)
                    Kolmogrov.append(mean_kst)

                    if mean_kst < 0.05:  # if drift
                        classifier = HoeffdingTreeClassifier()
                        classifier.fit(X_L2, Y_L2, np.unique(Y))
                        X_L1 = X_L2.copy()
                        Y_L1 = Y_L2.copy()
                    else:  # if no drift
                        [
                            New_X_Labeled,
                            New_Y_Labeled,
                        ] = Appending_informative_to_nextchunk(
                            X_L2, Y_L2, Informatives, Y_Informatives
                        )
                        classifier.partial_fit(
                            New_X_Labeled, New_Y_Labeled, np.unique(Y)
                        )
                        X_L1 = New_X_Labeled.copy()
                        Y_L1 = New_Y_Labeled.copy()
                else:
                    [New_X_Labeled, New_Y_Labeled] = Appending_informative_to_nextchunk(
                        X_L2, Y_L2, Informatives, Y_Informatives
                    )
                    classifier.partial_fit(
                        New_X_Labeled,
                        New_Y_Labeled,
                        np.unique(Y)
                    )
                    X_L1 = New_X_Labeled.copy()
                    Y_L1 = New_Y_Labeled.copy()
                X_cal1 = X_cal2.copy()
                Y_cal1 = Y_cal2.copy()
                X_U1 = X_U2.copy()
                Y_pred = classifier.predict(X_test)

                elapsed_time = time() - start

                hits = confusion_matrix(Y_test, Y_pred)
                _evaluate_metrics(Y_test, Y_pred)

                _log_iteration_info(
                    sum(hits.diagonal()),
                    stream.sample_idx,
                    elapsed_time,
                    mean_kst < 0.05
                )

                n_samples += chunk_size
                i += 1
            print(f'Finish {dataset} processing!!')
