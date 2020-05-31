import pandas as pd

import src.utils as utils

from scipy.sparse import csr_matrix, hstack

from src.core.states import RunningState
from .base import Callback, CallbackOrder


# on_split_start
class CheckDataFrameCallback(Callback):
    signature = "split"
    calback_order = CallbackOrder.ASSERTION

    def on_split_start(self, state: RunningState):
        if state.features.get("main") is None:
            msg = "main DataFrame does not exist. Aborting."
            raise KeyError(msg)

        if state.features["main"].get("train") is None:
            msg = "train DataFrame does not exist. Aborting."
            raise KeyError(msg)

        if not isinstance(state.features["main"]["train"], pd.DataFrame):
            if not isinstance(state.features["main"]["train"], csr_matrix):
                msg = "feature 'train' must be DataFrame"
                msg += " or a sparse matrix (csr). Aborting."
                raise ValueError(msg)


class PrepareForStratifiedKFoldCallback(Callback):
    signature = "split"
    callback_order = CallbackOrder.MIDDLE

    def on_split_start(self, state: RunningState):
        train = state.features["main"]["train"]
        target = state.target

        train["y"] = target
        state.features["main"]["train"] = train

    def on_split_end(self, state: RunningState):
        train = state.features["main"]["train"]
        train = train.drop("y", axis=1)
        state.features["main"]["train"] = train


class ConcatenateFeatureCallback(Callback):
    signature = "split"
    callback_type = CallbackOrder.HIGHEST

    def on_split_start(self, state: RunningState):
        features = state.features

        as_sparse = False
        for feature in features.values():
            if isinstance(feature["train"], dict):
                as_sparse = True
                break

        main_feature = {}
        with utils.timer("Concatnating `main` features", state.logger):
            if as_sparse:
                for phase in ["train", "test"]:
                    sparse_matrices = []
                    for f in features.values():
                        if isinstance(f[phase], pd.DataFrame):
                            feature_values = csr_matrix(f[phase].values)
                            sparse_matrices.append(feature_values)
                        elif isinstance(f[phase], dict):
                            sparse_dict = f[phase]
                            for sp_mat in sparse_dict.values():
                                sparse_matrices.append(sp_mat)
                    main_feature[phase] = hstack(sparse_matrices).tocsr()
            else:
                for phase in ["train", "test"]:
                    dfs = []
                    for f in features.values():
                        dfs.append(f[phase])

                    main_feature[phase] = pd.concat(dfs, axis=1)
        state.features["main"] = main_feature
