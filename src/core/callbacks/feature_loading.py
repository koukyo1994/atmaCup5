import pandas as pd

import src.utils as utils

from scipy.sparse import hstack, csr_matrix

from src.core.callbacks import Callback, CallbackOrder
from src.core.states import RunningState


class SortColumnsCallback(Callback):
    signature = "feature_loading"
    callback_order = CallbackOrder.MIDDLE

    def on_feature_loading_end(self, state: RunningState):
        features = state.features

        for key in features:
            if isinstance(features[key]["train"], pd.DataFrame):
                features[key]["train"] = features[key]["train"].sort_index(
                    axis=1)
                features[key]["test"] = features[key]["test"].sort_index(
                    axis=1)

        state.features = features


class ConcatenateFeatureCallback(Callback):
    signature = "feature_loading"
    callback_type = CallbackOrder.LOWER

    def on_feature_loading_end(self, state: RunningState):
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
