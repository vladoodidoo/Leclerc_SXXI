from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, PrecisionRecallDisplay

from sklearn.svm import SVC

import numpy as np

from preprocessing import *


def evaluate(expected, got):
    accuracy = accuracy_score(expected, got)
    f1 = f1_score(expected, got)
    recall = recall_score(expected, got)
    precision = precision_score(expected, got)
    mcc = matthews_corrcoef(expected, got)

    PrecisionRecallDisplay.from_predictions(expected, got)
    ConfusionMatrixDisplay(confusion_matrix(expected, got)).plot()

    return {"accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "matthews_corcoef": mcc}


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


def svc(total_df, cols):
    X = preprocess_df(total_df[cols])
    y = total_df[["label_n", "label"]]

    X_train, y_train = get_n_balanced_df(X, y, 500, label="label", label_n="label_n")

    model = SVC()
    model.fit(X_train, y_train)

    return {
        "training_metrics": evaluate(y_train, model.predict(X_train)),
        "testing_metrics": evaluate(y["label_n"], model.predict(X))
    }

def svc_physical(phy_df, x_test, y_test):
    pos_only = phy_df[phy_df["Label_n"] == 1].iloc[:500]
    neg_only = phy_df[phy_df["Label_n"] == 0].iloc[:500]

    training = pd.concat([pos_only, neg_only])

    X_train = training.loc[:, training.columns != "Label_n"]
    y_train = training["Label_n"]

    model = SVC()
    model.fit(X_train, y_train)

    return {
        "training_metrics": evaluate(y_train, model.predict(X_train)),
        "testing_metrics": evaluate(y_test, model.predict(x_test))
    }

def model_fit(model, phy_df, x_test, y_test):
    pos_only = phy_df[phy_df["Label_n"] == 1].iloc[:500]
    neg_only = phy_df[phy_df["Label_n"] == 0].iloc[:500]

    training = pd.concat([pos_only, neg_only])

    X_train = training.loc[:, training.columns != "Label_n"]
    y_train = training["Label_n"]
    model.fit(X_train, y_train)

    return {
        "training_metrics": evaluate(y_train, model.predict(X_train)),
        "testing_metrics": evaluate(y_test, model.predict(x_test))
    }

def model_fit_network(model, cols, net_df):
    X = preprocess_df(net_df[cols])
    y = net_df[["label_n", "label"]]

    X_train, y_train = get_n_balanced_df(X, y, 500, label="label", label_n="label_n")
    model.fit(X_train, y_train)

    return {
        "training_metrics": evaluate(y_train, model.predict(X_train)),
        "testing_metrics": evaluate(y["label_n"], model.predict(X))
    }