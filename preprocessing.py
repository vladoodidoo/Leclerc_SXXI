import pandas as pd


def convert_physical_datasets(path):
    bad_format = pd.read_csv(path, encoding='utf-16')
    columns = bad_format.columns[0].split("\t")
    lines = [line.item().split("\t") for line in bad_format.iloc]
    return pd.DataFrame(columns=columns, data=lines)
