"""
Preprocessing functions for the data
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_network_csv(path_list) -> pd.DataFrame:
    """
    Load network csv file and parse dates
    """

    # concat all csv files
    df = None
    for path in path_list:
        current_df = pd.read_csv(path, parse_dates=["Time"], engine="pyarrow")
        # Strip column names

        current_df.columns = [s.strip() for s in current_df.columns]

        if df is None:
            df = current_df
        else:
            df = pd.concat([df, current_df], ignore_index=True)


    # Remove unnecessary columns and basic preprocessing
    columns_to_drop = [
        "mac_s",
        "mac_d",
        "ip_s",
        "ip_d",
        "sport", # Source port to be removed as it can be controlled by attacker.
        "modbus_response",
        "label_n",
    ]
    df.drop(columns=columns_to_drop, inplace=True)

    df["flags"] = df["flags"].astype("str")
    df["dport"] = df["dport"].astype("str")

    # drop label with two small proportion
    df = df[(df.label == "normal") | (df.label == "DoS") | (df.label == "MITM")]

    return df


def load_physical_csv(path_list) -> pd.DataFrame:
    """
    Load physical csv files and basic preprocessing

    Args:
        path_list (list): List of paths to csv files

    Returns:
        pd.DataFrame: Dataframe with preprocessed data
    """

    df = None
    # Concat all csv files
    for path in path_list:
        # Correct encoding
        bad_format = pd.read_csv(path, encoding="utf-16")
        columns = bad_format.columns[0].split("\t")
        lines = [line.item().split("\t") for line in bad_format.iloc]

        # Create new dataframe
        current_df = pd.DataFrame(columns=columns, data=lines)
        # Parse dates
        current_df.Time = pd.to_datetime(current_df.Time, dayfirst=True)

        if df is None:
            df = current_df
        else:
            df = pd.concat([df, current_df], ignore_index=True)


    # Remove unnecessary columns and typo in column name
    df.drop(columns=["Label_n", "Lable_n", "Flow_sensor_3"], inplace=True)

    # Replace typo in label
    df.label = df.label.replace("nomal", "normal")

    df = df[(df.label == "normal") | (df.label == "physical fault")]

    df.rename(columns={"Label": "label"}, inplace=True)
    # Encore boolean values
    val_and_pump = list(df.filter(regex="Val|Pump").columns)
    df[val_and_pump] = (
        df[val_and_pump].map(lambda x: False if x == "false" else True).astype(int)
    )

    # Convert object to numeric
    sensors = list(df.filter(regex="Tank|Flow").columns)
    df[sensors] = df[sensors].apply(pd.to_numeric)

    return df


def time_split(df: pd.DataFrame, split_size: float) -> pd.DataFrame:
    """
    Split dataframe into train and test set based on time

    Args:
        df (pd.DataFrame): Dataframe to split
        split_size (float): Size of test set

    Returns:
        X_train, X_test, y_train, y_test: Train and test sets
    """
    df = df.sort_values(by="Time")
    # Drop time column
    df.drop(columns=["Time"], inplace=True)

    X, y = df.drop(columns=["label"]), df["label"]

    # Split with shuffle=False to preserve time order
    return train_test_split(X, y, test_size=0.2, shuffle=False)


def generate_sample(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """
    Downsample dataframe in order to reduce computation time

    Args:
        df (pd.DataFrame): Dataframe to downsample
        sample_size (int): Size of sample

    Returns:
        pd.DataFrame: Downsampled dataframe
    """

    # Sort by time and sample every nth row
    df = df.sort_values(by="Time")
    # Keep every nth row
    df = df.iloc[:: df.shape[0] // sample_size, :]
    return df


def generate_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate binary labels from multiclass labels

    Args:
        df (pd.DataFrame): Dataframe with multiclass labels

    Returns:
        pd.DataFrame: Dataframe with binary labels 1 for anomaly, 0 for normal
    """

    df["binary_label"] = df.label.map(lambda x: 0 if x == "normal" else 1)

    return df
