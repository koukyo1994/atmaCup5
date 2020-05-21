import numpy as np
import pandas as pd

import src.utils as utils

from scipy.io import loadmat

from . import SubRunner
from src.core.states import RunningState


class FeatureLoadingRunner(SubRunner):
    signature = "feature_loading"

    def __init__(self, config: dict, state: RunningState):
        super().__init__(config, state)

        self.callbacks = []

    def run(self):
        self._run_callbacks(phase="start")

        feature_dir = self.state.feature_dir
        for config in self.state.config:
            features = config.get("features")
            for feature_dict in features:
                feature_name = feature_dict["name"]
                path = feature_dict["path"]
                role = feature_dict["role"]

                if role == "target":
                    with utils.timer(f"Load {str(feature_dir / path)}",
                                     self.state.logger):
                        self.state.target = np.load(feature_dir / path)
                    continue

                feature_format = path.split(".")[-1]

                if feature_name not in self.state.features.keys():
                    self.state.features[feature_name] = {}

                with utils.timer(f"Load {str(feature_dir / path)}",
                                 self.state.logger):
                    if feature_format == "mat":
                        self.state.features[feature_name][role] = loadmat(
                            feature_dir / path)[feature_name].tocsr()
                    elif feature_format in {"feather", "ftr"}:
                        self.state.features[feature_name][
                            role] = pd.read_feather(feature_dir / path)
                    else:
                        raise NotImplementedError

        self._run_callbacks(phase="end")
