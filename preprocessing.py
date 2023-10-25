import pandas as pd


def load_physical_csv(path):
    bad_format = pd.read_csv(path, encoding='utf-16')
    columns = bad_format.columns[0].split("\t")
    lines = [line.item().split("\t") for line in bad_format.iloc]
    df = pd.DataFrame(columns=columns, data=lines)
    return df.loc[:, (df.columns != 'Time') & (df.columns != 'Label')].astype({
        'Pump_1': bool,
        'Pump_2': bool,
        'Pump_3': bool,
        'Pump_4': bool,
        'Pump_5': bool,
        'Pump_6': bool,
        'Valv_1': bool,
        'Valv_2': bool,
        'Valv_3': bool,
        'Valv_4': bool,
        'Valv_5': bool,
        'Valv_6': bool,
        'Valv_7': bool,
        'Valv_8': bool,
        'Valv_9': bool,
        'Valv_10': bool,
        'Valv_11': bool,
        'Valv_12': bool,
        'Valv_13': bool,
        'Valv_14': bool,
        'Valv_15': bool,
        'Valv_16': bool,
        'Valv_17': bool,
        'Valv_18': bool,
        'Valv_19': bool,
        'Valv_20': bool,
        'Valv_21': bool,
        'Valv_22': bool,
    }).astype(int)


def load_network_csv(path):
    df = pd.read_csv(path)
    df.columns = [s.strip() for s in df.columns]
    df["flags"] = df["flags"].astype('str')
    df["dport"] = df["dport"].astype('str')
    return df


def get_one_hot_encoded_df(df):
    return pd.get_dummies(df)


def get_object_columns(df):
    return df.select_dtypes(include="object").columns.tolist()


def get_number_columns(df):
    return df.select_dtypes(include="number").columns.tolist()


def remove_nan_by_mean(df):
    return df.fillna(df.mean())


def preprocess_df(df):
    objects = get_object_columns(df)
    tmp1 = get_one_hot_encoded_df(df[objects])

    numbers = get_number_columns(df)
    tmp2 = remove_nan_by_mean(df[numbers])

    return pd.concat([tmp2, tmp1], axis=1)


def get_n_balanced_df(df, y, n, label_n='label_n', label='label'):
    df = pd.concat([df, y], axis=1)
    positive = df[(df[label_n] == 1) & (df[label] != 'anomaly')].iloc[:n]

    negative = df[df[label_n] == 0].iloc[:n]

    result = pd.concat([positive, negative])

    y = result[label_n]
    X = result.loc[:, (result.columns != label_n) & (result.columns != label)]

    return X, y
