import pandas as pd

from scipy.io import savemat
from scipy.sparse import csr_matrix

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
                if isinstance(feature[phase], csr_matrix):
                    mdict = {name: feature[phase]}
                    savemat(feature_dir / f"{name}_{phase}.mat", mdict)
                elif isinstance(feature[phase], pd.DataFrame):
                    feature.to_feather(feature_dir / f"{name}_{phase}.ftr")
