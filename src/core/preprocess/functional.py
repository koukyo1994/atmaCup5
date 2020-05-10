import numpy as np
import pandas as pd

from typing import List

from fastprogress import progress_bar


def find_constant_columns(df: pd.DataFrame) -> List[str]:
    return df.nunique().reset_index(
        name="nunique").query("nunique == 1")["index"].tolist()


def find_duplicated_columns(df: pd.DataFrame) -> List[str]:
    to_remove = []
    columns = df.columns.tolist()
    for i in range(len(columns) - 1):
        values = df[columns[i]].values
        for j in range(i + 1, len(columns)):
            if np.array_equal(values, df[columns[j]].values):
                to_remove.append(columns[j])
    return to_remove


def find_correlated_columns(df: pd.DataFrame, threshold=0.995) -> List[str]:
    to_remove: List[str] = []
    columns = df.columns.tolist()
    for i in progress_bar(range(len(columns) - 1), leave=True):
        column_a = columns[i]
        for j in range(i + 1, len(columns)):
            column_b = columns[j]
            if column_a in to_remove or column_b in to_remove:
                continue
            values_a = df[column_a].values
            values_b = df[column_b].values
            if abs(np.corrcoef(values_a, values_b)[0, 1]):
                to_remove.append(column_b)
    return to_remove
