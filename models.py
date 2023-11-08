"""
Models for anomaly detection and classification of attacks
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skrub import TableVectorizer


def evaluation_report(y_true: np.ndarray, y_preds: np.ndarray) -> dict:
    """
    Generate evaluation report

    Args:
        y_true (np.ndarray): True labels
        y_preds (np.ndarray): Predicted labels

    Returns:
        dict: Evaluation report
    """

    accuracy = accuracy_score(y_true, y_preds)
    balanced_accuracy = balanced_accuracy_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds, average="weighted")
    recall = recall_score(y_true, y_preds, average="weighted")
    precision = precision_score(y_true, y_preds, average="weighted")
    mcc = matthews_corrcoef(y_true, y_preds)

    return {
        "accuracy_score": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "matthews_corcoef": mcc,
    }


def pretty_print_report(report: dict) -> None:
    """
    Pretty print evaluation report

    Args:
        report (dict): Evaluation report
    """

    print(
        f"""
    Accuracy:           {report['accuracy_score']}
    Balanced Accuracy:  {report['balanced_accuracy']}
    Precision:          {report['precision']}
    Recall:             {report['recall']}
    F1 Score:           {report['f1_score']}
    MCC:                {report['matthews_corcoef']}
    """
    )


def evaluation_plot(
    y_true: np.ndarray, y_preds: np.ndarray, classes: list
) -> None:
    """
    Plot confusion matrix and Roc curve

    Args:
        y_true (np.ndarray): True labels
        y_preds (np.ndarray): Predicted labels
        classes (list): List of classes
    """
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_preds), display_labels=classes
    ).plot(cmap="Blues", xticks_rotation="vertical")


def generate_pipeline():
    """
    Generate pipeline for model

    Returns:
        pipeline: Pipeline for model
    """
    pipeline = make_pipeline(
        TableVectorizer(sparse_threshold=0.0),
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )

    return pipeline


def lof(X_test: np.ndarray, contamination: float) -> np.ndarray:
    """
    Local Outlier Factor model for anomaly detection

    Args:
        df (pd.DataFrame): Dataframe to run anomaly detection on
        contamination (float): Percentage of outliers

    Returns:
        np.ndarray: Array of 1s and 0s, 1 being an outlier
    """

    pipeline = make_pipeline(
        TableVectorizer(),
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LocalOutlierFactor(contamination=contamination, n_jobs=-1),
    )

    y_preds = np.array(pipeline.fit_predict(X_test))

    return np.where(y_preds == -1, 1, 0)


def isolation_forest(X_test: np.ndarray, contamination: float) -> np.ndarray:
    """
    Isolation Forest model for anomaly detection

    Args:
        df (pd.DataFrame): Dataframe to run anomaly detection on
        contamination (float): Percentage of outliers

    Returns:
        np.ndarray: Array of 1s and 0s, 1 being an outlier
    """

    pipeline = make_pipeline(
        TableVectorizer(),
        SimpleImputer(strategy="median"),
        StandardScaler(),
        IsolationForest(random_state=42, contamination=contamination),
    )

    y_preds = np.array(pipeline.fit_predict(X_test))

    return np.where(y_preds == -1, 1, 0)
