import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
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


def plot_confusion_matrix(
    y_true: np.ndarray, y_preds: np.ndarray, classes: list
) -> None:
    """
    Plot confusion matrix

    Args:
        y_true (np.ndarray): True labels
        y_preds (np.ndarray): Predicted labels
        classes (list): List of classes
    """
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_preds), display_labels=classes
    ).plot(cmap="Blues", xticks_rotation="vertical")


def generate_pipeline(model, X_train, y_train):
    """
    Generate pipeline for model

    Args:
        model (sklearn model): Model to generate pipeline for
        X_train (pd.DataFrame): Training data
        y_train (pd.DataFrame): Training labels

    Returns:
        pipeline: Pipeline for model
    """
    pipeline = make_pipeline(
        TableVectorizer(), SimpleImputer(strategy="median"), StandardScaler(), model
    )
    pipeline.fit(X_train, y_train)
    return pipeline


# def lof(df, contamination, preprocess=True):
#     if preprocess:
#         df = preprocess_df(df)
#     model = LocalOutlierFactor(contamination=contamination, n_jobs=-1)
#     y = np.array(model.fit_predict(df))

#     return np.where(y == -1, 1, 0)


# def isolation_forest(df, contamination, preprocess=True):
#     if preprocess:
#         df = preprocess_df(df)
#     model = IsolationForest(random_state=42, contamination=contamination)
#     y = np.array(model.fit_predict(df))

#     return np.where(y == -1, 1, 0)


# def svc(total_df, cols):
#     X = preprocess_df(total_df[cols])
#     y = total_df[["label_n", "label"]]

#     X_train, y_train = get_n_balanced_df(X, y, 500, label="label", label_n="label_n")

#     model = SVC()
#     model.fit(X_train, y_train)

#     return {
#         "training_metrics": evaluate(y_train, model.predict(X_train)),
#         "testing_metrics": evaluate(y["label_n"], model.predict(X))
#     }

# def svc_physical(phy_df, x_test, y_test):
#     pos_only = phy_df[phy_df["Label_n"] == 1].iloc[:500]
#     neg_only = phy_df[phy_df["Label_n"] == 0].iloc[:500]

#     training = pd.concat([pos_only, neg_only])

#     X_train = training.loc[:, training.columns != "Label_n"]
#     y_train = training["Label_n"]

#     model = SVC()
#     model.fit(X_train, y_train)

#     return {
#         "training_metrics": evaluate(y_train, model.predict(X_train)),
#         "testing_metrics": evaluate(y_test, model.predict(x_test))
#     }

# def model_fit(model, phy_df, x_test, y_test):
#     pos_only = phy_df[phy_df["Label_n"] == 1].iloc[:500]
#     neg_only = phy_df[phy_df["Label_n"] == 0].iloc[:500]

#     training = pd.concat([pos_only, neg_only])

#     X_train = training.loc[:, training.columns != "Label_n"]
#     y_train = training["Label_n"]
#     model.fit(X_train, y_train)

#     return {
#         "training_metrics": evaluate(y_train, model.predict(X_train)),
#         "testing_metrics": evaluate(y_test, model.predict(x_test))
#     }
