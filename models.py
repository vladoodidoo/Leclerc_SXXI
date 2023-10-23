from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import numpy as np

from preprocessing import *


def evaluate(expected, got):
    accuracy = accuracy_score(expected, got)
    f1 = f1_score(expected, got)
    recall = recall_score(expected, got)
    precision = precision_score(expected, got)

    return {"accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1}


def lof(df, contamination, preprocess=True):
    if preprocess:
        df = preprocess_df(df)
    model = LocalOutlierFactor(contamination=contamination)
    y = np.array(model.fit_predict(df))

    return np.where(y == -1, 1, 0)


def isolation_forest(df, contamination, preprocess=True):
    if preprocess:
        df = preprocess_df(df)
    model = IsolationForest(random_state=42, contamination=contamination)
    y = np.array(model.fit_predict(df))

    return np.where(y == -1, 1, 0)


def svc(total_df, cols, y=None, preprocess=True):
    if preprocess:
        X = preprocess_df(total_df[cols])
        y = total_df[["label_n", "label"]]
    else:
        X = total_df

    X_train, y_train = get_n_balanced_df(X, y, 500)

    model = SVC()
    model.fit(X_train, y_train)

    return {
        "training_metrics": evaluate(y_train, model.predict(X_train))
    }

