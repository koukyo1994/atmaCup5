import numpy as np
import pandas as pd

from typing import List, Tuple

from sklearn import model_selection


def get_split(df: pd.DataFrame,
              config: dict) -> List[Tuple[np.ndarray, np.ndarray]]:
    name: str = config["how"]
    func = globals().get(name)
    if func is None:
        raise NotImplementedError

    return func(df, config)


def kfold(df: pd.DataFrame,
          config: dict) -> List[Tuple[np.ndarray, np.ndarray]]:
    params = config["params"]
    splitter = model_selection.KFold(**params)
    return list(splitter.split(df))


def train_test_split(df: pd.DataFrame,
                     config: dict) -> List[Tuple[np.ndarray, np.ndarray]]:
    params = config["params"]
    index = np.arange(len(df))
    return [model_selection.train_test_split(index, **params)]


def group_kfold(df: pd.DataFrame,
                config: dict) -> List[Tuple[np.ndarray, np.ndarray]]:
    params = config["params"]
    group = df["group"]
    splitter = model_selection.GroupKFold(**params)
    return list(splitter.split(df, groups=group))


def random_group_kfold(df: pd.DataFrame,
                       config: dict) -> List[Tuple[np.ndarray, np.ndarray]]:
    params = config["params"]
    group = df["group"]
    splitter = model_selection.KFold(**params)
    uniq_groups = group.unique()

    split = []
    for trn_grp_idx, val_grp_idx in splitter.split(uniq_groups):
        trn_grp = uniq_groups[trn_grp_idx]
        val_grp = uniq_groups[val_grp_idx]
        trn_idx = df[df["group"].isin(trn_grp)].index.values
        val_idx = df[df["group"].isin(val_grp)].index.values
        split.append((trn_idx, val_idx))
    return split
