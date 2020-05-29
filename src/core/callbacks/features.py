import gc

import numpy as np
import pandas as pd

import src.utils as utils

from typing import Optional

from scipy.io import savemat
from scipy.sparse import csr_matrix, hstack

from src.core.states import RunningState

from .base import Callback, CallbackOrder


# on_features_start
class AssignTargetCallback(Callback):
    signature = "features"
    callback_order = CallbackOrder.LOWEST

    def on_features_start(self, state: RunningState):
        dataframes = state.dataframes
        for key, df in dataframes.items():
            if state.target_name in df.columns:
                state.logger.info(
                    f"Found target `{state.target_name}` in {key}")
                state.target = df[state.target_name].values
                break


# on_features_end
class FeatureSavingCallback(Callback):
    signature = "features"
    callback_order = CallbackOrder.LOWEST

    def __init__(self, prefix: Optional[str] = None):
        if prefix is None:
            self.prefix = ""
        else:
            self.prefix = prefix + "_"

    def on_features_end(self, state: RunningState):
        feature_dir = state.feature_dir
        features = state.features

        for name, feature in features.items():
            for phase in ["train", "test"]:
                if isinstance(feature[phase], dict) or isinstance(
                        feature[phase], csr_matrix):
                    mdict = {name: feature[phase]}
                    with utils.timer(
                            "Saving " + f"{self.prefix}{name}_{phase}.mat",
                            state.logger):
                        savemat(
                            feature_dir / f"{self.prefix}{name}_{phase}.mat",
                            mdict)
                elif isinstance(feature[phase], pd.DataFrame):
                    with utils.timer(
                            "Saving " + f"{self.prefix}{name}_{phase}.ftr",
                            state.logger):
                        for col in feature[phase].columns:
                            if feature[phase][col].dtype == "float16":
                                feature[phase][col] = feature[phase][
                                    col].astype("float32")
                        feature[phase].to_feather(
                            feature_dir / f"{self.prefix}{name}_{phase}.ftr")
                else:
                    raise NotImplementedError

        target = state.target
        with utils.timer("Saving main_target.npy", state.logger):
            np.save(feature_dir / f"{self.prefix}main_target.npy", target)


class ConcatenateFeatureCallback(Callback):
    signature = "features"
    callback_order = CallbackOrder.HIGHEST

    def on_features_end(self, state: RunningState):
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

        keys = list(features.keys())
        keys.remove("main")

        for key in keys:
            del state.features[key]
            gc.collect()
