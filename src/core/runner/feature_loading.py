import pandas as pd

from scipy.io import loadmat

from . import SubRunner
from src.core.stats import RunningState


class FeatureLoadingRunner(SubRunner):
    signature = "feature_loading"

    def __init__(self, config: dict, state: RunningState):
        super().__init__(config, state)

        self.callbacks = []

    def run(self):
        self._run_callbacks(phase="start")

        # TODO: apply role
        feature_dir = self.state.feature_dir
        for config in self.state.config:
            features = config.get("features")
            for feature_dict in features:
                feature_name = feature_dict["name"]
                path = feature_dict["path"]

                feature_format = path.split(".")[-1]
                if feature_format == ".mat":
                    self.state.features[feature_name] = loadmat(
                        feature_dir / path)[feature_name]
                elif feature_format in {"feather", "ftr"}:
                    self.state.features[feature_name] = pd.read_feather(
                        feature_dir / path)
                else:
                    raise NotImplementedError

        self._run_callbacks(phase="end")
