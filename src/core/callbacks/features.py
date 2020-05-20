import pandas as pd

from scipy.io import savemat
from scipy.sparse import csr_matrix

from src.core.state import RunningState

from .base import Callback, CallbackOrder


# on_features_end
class FeatureSaving(Callback):
    signature = "features"
    callback_order = CallbackOrder.LOWEST

    def on_features_end(self, state: RunningState):
        feature_dir = state.feature_dir
        features = state.features

        for name, feature in features.items():
            if isinstance(feature, csr_matrix):
                mdict = {name: feature}
                savemat(feature_dir / f"{name}.mat", mdict)
            elif isinstance(feature, pd.DataFrame):
                feature.to_feather(feature_dir / f"{name}.ftr")
