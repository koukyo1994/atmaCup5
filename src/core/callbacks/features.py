import gc

import pandas as pd

import src.utils as utils

from scipy.io import savemat
from scipy.sparse import csr_matrix, hstack

from src.core.states import RunningState

from .base import Callback, CallbackOrder


# on_features_end
class FeatureSavingCallback(Callback):
    signature = "features"
    callback_order = CallbackOrder.LOWEST

    def on_features_end(self, state: RunningState):
        feature_dir = state.feature_dir
        features = state.features

        for name, feature in features.items():
            for phase in ["train", "test"]:
                if isinstance(feature[phase], dict):
                    mdict = {name: feature[phase]}
                    with utils.timer("Saving " + f"{name}_{phase}.mat",
                                     state.logger):
                        savemat(feature_dir / f"{name}_{phase}.mat", mdict)
                elif isinstance(feature[phase], pd.DataFrame):
                    with utils.timer("Saving " + f"{name}_{phase}.ftr",
                                     state.logger):
                        feature[phase].to_feather(
                            feature_dir / f"{name}_{phase}.ftr")
                else:
                    raise NotImplementedError


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
                    main_feature[phase] = {"main": hstack(sparse_matrices)}
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
