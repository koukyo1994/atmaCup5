import src.utils as utils

from src.core.callbacks import Callback, CallbackOrder
from src.core.states import RunningState


# on_features_start
class MergeFittingCallback(Callback):
    signature = "features"
    callback_order = CallbackOrder.LOWER

    def on_features_start(self, state: RunningState):
        fitting_key = "input/atma5/fitting.csv"
        fitting = state.dataframes[fitting_key]
        for phase in ["train", "test"]:
            key = f"input/atma5/{phase}.csv"
            msg = f"Merging `{key}` with fitting.csv"
            with utils.timer(msg, state.logger):
                df = state.dataframes[key]
                state.dataframes[key] = df.merge(
                    fitting, how="left", on="spectrum_id")
