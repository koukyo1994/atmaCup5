import pandas as pd

from pathlib import Path
from typing import Dict, List, Union

from fastprogress import progress_bar
from tsfresh import extract_relevant_features, extract_features


class BasicOperationsOnSpectrum:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_transform(self, X: pd.DataFrame):
        return self.transform(X)

    def transform(self, X: pd.DataFrame):
        operations = self.kwargs["operations"]
        prefix = self.kwargs["prefix"] + "_"
        result_dict: Dict[str, List[Union[int, float]]] = {
            op: []
            for op in operations
        }

        base_dir = Path("input/atma5/spectrum")
        for _, row in progress_bar(X.iterrows(), total=len(X)):
            spectrum = pd.read_csv(
                base_dir / row.spectrum_filename, sep="\t", header=None)
            for op in operations:
                result_dict[op].append(spectrum[1].__getattribute__(op)())
        df = pd.DataFrame(result_dict)
        df.columns = [prefix + c for c in df.columns]
        return df


class TsfreshRelevantFeatures:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.parameters = {}

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        X.wl = X.wl.astype("float32")
        X.intensity = X.intensity.astype("float32")
        y.index = X.spectrum_filename.unique()
        features = extract_relevant_features(X, y, **self.kwargs)

        columns = features.columns
        for column in columns:
            params = column.split("__")

            extractor_name = params[1]
            if len(params) == 2:
                extractor_params = None
            else:
                extractor_params = params[2:]

            if extractor_name not in self.parameters.keys():
                if extractor_params is None:
                    self.parameters[extractor_name] = None
                else:
                    self.parameters[extractor_name] = []
                    parameter = {}
                    for param in extractor_params:
                        param_var = param.split("_")
                        if len(param_var) > 2:
                            param_name = "_".join(param_var[:-1])
                            param_value = param_var[-1]
                        else:
                            param_name = param_var[0]
                            param_value = param_var[1]
                        parameter[param_name] = eval(param_value)

                    self.parameters[extractor_name].append(parameter)
            else:
                if extractor_params is not None:
                    parameter = {}
                    for param in extractor_params:
                        param_var = param.split("_")
                        if len(param_var) > 2:
                            param_name = "_".join(param_var[:-1])
                            param_value = param_var[-1]
                        else:
                            param_name = param_var[0]
                            param_value = param_var[1]
                        parameter[param_name] = eval(param_value)

                    self.parameters[extractor_name].append(parameter)

        return features.reset_index(drop=True)

    def transform(self, X: pd.DataFrame):
        X.wl = X.wl.astype("float32")
        X.intensity = X.intensity.astype("float32")
        features = extract_features(
            X, default_fc_parameters=self.parameters, **self.kwargs)
        return features.reset_index(drop=True)
