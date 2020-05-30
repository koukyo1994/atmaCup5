import pandas as pd

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
