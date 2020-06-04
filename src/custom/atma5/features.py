import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict, List, Union

from fastprogress import progress_bar
from scipy.interpolate import UnivariateSpline
from scipy import integrate
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


class SpectrumIntegral:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_transform(self, X: pd.DataFrame):
        return self.transform(X)

    def transform(self, X: pd.DataFrame):
        unique_filenames = X["spectrum_filename"].unique()

        integrals = []
        for filename in progress_bar(unique_filenames):
            spec = X.query(f"spectrum_filename == '{filename}'")

            x = spec["wl"].values
            y = spec["intensity"].values

            method = self.kwargs["how"]
            integrals.append(integrate.__getattribute__(method)(y, x))
        return pd.DataFrame({"spectrum_integral": integrals})


class SpectrumPeakFeatures:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_transform(self, X: pd.DataFrame):
        return self.transform(X)

    def transform(self, X: pd.DataFrame):
        unique_filenames = X["spectrum_filename"].unique()
        features = {}

        diff_spans = self.kwargs["diff_span"]
        for s in diff_spans:
            features[f"max_steapness_span_{s}"] = np.zeros(
                len(unique_filenames))
            features[f"highest_peak_steapness_span_{s}"] = np.zeros(
                len(unique_filenames))

        for i, filename in enumerate(progress_bar(unique_filenames)):
            spec = X.query(f"spectrum_filename == '{filename}'")

            x = spec["wl"].values
            y = spec["intensity"].values

            spline = UnivariateSpline(x, y - np.max(y) * 0.4, s=0)
            roots = spline.roots()
            if len(roots) < 2:
                continue

            steapness_lists: Dict[int, list] = {s: [] for s in diff_spans}
            peak_heights = []
            for j in range(len(roots) // 2):
                left = roots[j * 2]
                right = roots[j * 2 + 1]

                span_indices = np.logical_and(x <= right, x >= left)
                if span_indices.mean() == 0:
                    peak_wl = (left + right) / 2.0
                    abs_dist = np.abs(x - peak_wl)
                    peak_index = np.argmin(abs_dist)
                else:
                    peak_index_in_span = y[span_indices].argmax()
                    peak_wl = x[span_indices][peak_index_in_span]
                    peak_index = x.tolist().index(peak_wl)

                if peak_index == 0 or peak_index == (len(x) - 1):
                    continue

                peak_heights.append(y[peak_index])
                for s in diff_spans:
                    left_span_start = peak_index - s
                    if left_span_start < 0:
                        left_span_start = 0

                    right_span_end = peak_index + s
                    if right_span_end > len(x) - 1:
                        right_span_end = len(x) - 1

                    left_diff = (y[peak_index] - y[left_span_start]) / (
                        x[peak_index] - x[left_span_start])
                    right_diff = (y[peak_index] - y[right_span_end]) / (
                        x[right_span_end] - x[peak_index])

                    steapness_lists[s].append(max(left_diff, right_diff))

            for s in diff_spans:
                steapness = steapness_lists[s]

                features[f"max_steapness_span_{s}"][i] = np.max(steapness)

                if len(peak_heights) == 0:
                    features[f"highest_peak_steapness_span_{s}"][
                        i] = np.random.choice(steapness)
                else:
                    highest_peak_height = np.max(peak_heights)
                    highest_peak_idx = peak_heights.index(highest_peak_height)
                    features[f"highest_peak_steapness_span_{s}"][
                        i] = steapness[highest_peak_idx]

        return pd.DataFrame(features)


class NormalizedPeakFeatures:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_transform(self, X: pd.DataFrame):
        return self.transform(X)

    def transform(self, X: pd.DataFrame):
        unique_filenames = X["spectrum_filename"].unique()
        features = {}
        eps = 1e-7

        widths = self.kwargs["widths"]
        for w in widths:
            features[f"max_inverse_fw_{w}"] = np.zeros(len(unique_filenames))
            features[f"highest_peak_inverse_fw_{w}"] = np.zeros(
                len(unique_filenames))
            features[f"max_inverse_hw_{w}"] = np.zeros(len(unique_filenames))
            features[f"highest_peak_inverse_hw_{w}"] = np.zeros(
                len(unique_filenames))

        diff_spans = self.kwargs["diff_span"]
        for s in diff_spans:
            features[f"max_steapness_span_{s}"] = np.zeros(
                len(unique_filenames))
            features[f"highest_peak_steapness_span_{s}"] = np.zeros(
                len(unique_filenames))

        for i, filename in enumerate(progress_bar(unique_filenames)):
            spec = X.query(f"spectrum_filename == '{filename}'")

            x = spec["wl"].values
            y = spec["intensity"].values

            y = (y - y.min()) / (y.max() - y.min())
            spline = UnivariateSpline(x, y - np.max(y) * 0.4, s=0)
            roots = spline.roots()
            if len(roots) < 2:
                continue

            steapness_lists: Dict[int, list] = {s: [] for s in diff_spans}
            peak_heights = []
            for j in range(len(roots) // 2):
                left = roots[j * 2]
                right = roots[j * 2 + 1]

                span_indices = np.logical_and(x <= right, x >= left)
                if span_indices.mean() == 0:
                    peak_wl = (left + right) / 2.0
                    abs_dist = np.abs(x - peak_wl)
                    peak_index = np.argmin(abs_dist)
                else:
                    peak_index_in_span = y[span_indices].argmax()
                    peak_wl = x[span_indices][peak_index_in_span]
                    peak_index = x.tolist().index(peak_wl)

                if peak_index == 0 or peak_index == (len(x) - 1):
                    continue

                peak_heights.append(y[peak_index])
                for s in diff_spans:
                    left_span_start = peak_index - s
                    if left_span_start < 0:
                        left_span_start = 0

                    right_span_end = peak_index + s
                    if right_span_end > len(x) - 1:
                        right_span_end = len(x) - 1

                    left_diff = (y[peak_index] - y[left_span_start]) / (
                        x[peak_index] - x[left_span_start])
                    right_diff = (y[peak_index] - y[right_span_end]) / (
                        x[right_span_end] - x[peak_index])

                    steapness_lists[s].append(max(left_diff, right_diff))

            for s in diff_spans:
                steapness = steapness_lists[s]
                if len(steapness) != 0:
                    features[f"max_steapness_span_{s}"][i] = np.max(steapness)

                if len(peak_heights) == 0:
                    if len(steapness) != 0:
                        features[f"highest_peak_steapness_span_{s}"][
                            i] = np.random.choice(steapness)
                else:
                    highest_peak_height = np.max(peak_heights)
                    highest_peak_idx = peak_heights.index(highest_peak_height)
                    if len(steapness) != 0:
                        features[f"highest_peak_steapness_span_{s}"][
                            i] = steapness[highest_peak_idx]

            for w in widths:
                spline = UnivariateSpline(x, y - np.max(y) * w, s=0)
                roots = spline.roots()
                if len(roots) < 2:
                    continue

                inverse_fwhms = []
                inverse_hwhms = []
                peak_heights = []
                for j in range(len(roots) // 2):
                    left = roots[j * 2]
                    right = roots[j * 2 + 1]

                    inverse_fwhms.append(1.0 / (right - left))

                    span_indices = np.logical_and(x <= roots[j * 2 + 1],
                                                  x >= roots[j * 2])
                    if span_indices.mean() == 0.0:
                        peak_wl = (left + right) / 2.0
                        abs_dist = np.abs(x - peak_wl)
                        peak_index = np.argmin(abs_dist)
                    else:
                        peak_index_in_span = y[span_indices].argmax()
                        peak_wl = x[span_indices][peak_index_in_span]
                        peak_index = x.tolist().index(peak_wl)
                    peak_heights.append(y[peak_index])

                    inverse_left_hwhm = 1.0 / (peak_wl + eps - left)
                    inverse_right_hwhm = 1.0 / (right + eps - peak_wl)

                    inverse_hwhms.append(
                        max(inverse_left_hwhm, inverse_right_hwhm))

                features[f"max_inverse_fw_{w}"][i] = np.max(inverse_fwhms)
                features[f"max_inverse_hw_{w}"][i] = np.max(inverse_hwhms)

                if len(peak_heights) == 0:
                    features[f"highest_peak_inverse_fw_{w}"][
                        i] = np.random.choice(inverse_fwhms)
                    features[f"highest_peak_inverse_hw_{w}"][
                        i] = np.random.choice(inverse_hwhms)
                else:
                    highest_peak_height = np.max(peak_heights)
                    highest_peak_idx = peak_heights.index(highest_peak_height)
                    features[f"highest_peak_inverse_fw_{w}"][
                        i] = inverse_fwhms[highest_peak_idx]
                    features[f"highest_peak_inverse_hw_{w}"][
                        i] = inverse_hwhms[highest_peak_idx]

        return pd.DataFrame(features)


class ParametrizedFWHM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_transform(self, X: pd.DataFrame):
        return self.transform(X)

    def transform(self, X: pd.DataFrame):
        unique_filenames = X["spectrum_filename"].unique()
        features = {}
        eps = 1e-7

        widths = self.kwargs["width"]
        for w in widths:
            features[f"max_inverse_fw_{w}"] = np.zeros(len(unique_filenames))
            features[f"highest_peak_inverse_fw_{w}"] = np.zeros(
                len(unique_filenames))
            features[f"max_inverse_hw_{w}"] = np.zeros(len(unique_filenames))
            features[f"highest_peak_inverse_hw_{w}"] = np.zeros(
                len(unique_filenames))

        for i, filename in enumerate(progress_bar(unique_filenames)):
            spec = X.query(f"spectrum_filename == '{filename}'")

            x = spec["wl"].values
            y = spec["intensity"].values

            for w in widths:
                spline = UnivariateSpline(x, y - np.max(y) * w, s=0)
                roots = spline.roots()
                if len(roots) < 2:
                    continue

                inverse_fwhms = []
                inverse_hwhms = []
                peak_heights = []
                for j in range(len(roots) // 2):
                    left = roots[j * 2]
                    right = roots[j * 2 + 1]

                    inverse_fwhms.append(1.0 / (right - left))

                    span_indices = np.logical_and(x <= roots[j * 2 + 1],
                                                  x >= roots[j * 2])
                    if span_indices.mean() == 0.0:
                        peak_wl = (left + right) / 2.0
                        abs_dist = np.abs(x - peak_wl)
                        peak_index = np.argmin(abs_dist)
                    else:
                        peak_index_in_span = y[span_indices].argmax()
                        peak_wl = x[span_indices][peak_index_in_span]
                        peak_index = x.tolist().index(peak_wl)
                    peak_heights.append(y[peak_index])

                    inverse_left_hwhm = 1.0 / (peak_wl + eps - left)
                    inverse_right_hwhm = 1.0 / (right + eps - peak_wl)

                    inverse_hwhms.append(
                        max(inverse_left_hwhm, inverse_right_hwhm))

                features[f"max_inverse_fw_{w}"][i] = np.max(inverse_fwhms)
                features[f"max_inverse_hw_{w}"][i] = np.max(inverse_hwhms)

                if len(peak_heights) == 0:
                    features[f"highest_peak_inverse_fw_{w}"][
                        i] = np.random.choice(inverse_fwhms)
                    features[f"highest_peak_inverse_hw_{w}"][
                        i] = np.random.choice(inverse_hwhms)
                else:
                    highest_peak_height = np.max(peak_heights)
                    highest_peak_idx = peak_heights.index(highest_peak_height)
                    features[f"highest_peak_inverse_fw_{w}"][
                        i] = inverse_fwhms[highest_peak_idx]
                    features[f"highest_peak_inverse_hw_{w}"][
                        i] = inverse_hwhms[highest_peak_idx]

        return pd.DataFrame(features)


class SpectrumAdvancedFeatures:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_transform(self, X: pd.DataFrame):
        return self.transform(X)

    def transform(self, X: pd.DataFrame):
        unique_filenames = X["spectrum_filename"].unique()

        max_fwhm_inverse = []
        mean_fwhm_inverse = []
        num_peaks = []
        highest_peak_fwhm_inverse = []
        highest_peak_positions = []
        for filename in progress_bar(unique_filenames):
            spec = X.query(f"spectrum_filename == '{filename}'")

            x = spec["wl"].values
            y = spec["intensity"].values

            highest_peak_pos = x[y.argmax()]
            highest_peak_positions.append(highest_peak_pos)

            spline = UnivariateSpline(x, y - np.max(y) / 2, s=0)
            roots = spline.roots()
            if len(roots) < 2:
                max_fwhm_inverse.append(0)
                mean_fwhm_inverse.append(0)
                highest_peak_fwhm_inverse.append(0)

                if len(roots) == 1:
                    num_peaks.append(1)
                else:
                    num_peaks.append(0)
                continue
            inverse_fwhms = []
            for i in range(len(roots) // 2):
                left = roots[i * 2]
                right = roots[i * 2 + 1]
                inverse_fwhms.append(1.0 / (right - left))

            max_fwhm_inverse.append(np.max(inverse_fwhms))
            mean_fwhm_inverse.append(np.mean(inverse_fwhms))
            num_peaks.append(len(roots) // 2)

            try:
                highest_peak_index = np.argwhere(
                    roots < highest_peak_pos).max() // 2
                highest_peak_fwhm_inverse.append(
                    inverse_fwhms[highest_peak_index])
            except IndexError:
                highest_peak_fwhm_inverse.append(
                    np.random.choice(inverse_fwhms))
            except ValueError:
                highest_peak_fwhm_inverse.append(
                    np.random.choice(inverse_fwhms))
        return pd.DataFrame({
            "max_fwhm_inverse":
            max_fwhm_inverse,
            "mean_fwhm_inverse":
            mean_fwhm_inverse,
            "num_peaks":
            num_peaks,
            "highest_peak_fwhm_inverse":
            highest_peak_fwhm_inverse,
            "highest_peak_positions":
            highest_peak_positions
        })


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


class InverseSpecMax:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_transform(self, X: pd.DataFrame):
        return self.transform(X)

    def transform(self, X: pd.DataFrame):
        prefix = self.kwargs["prefix"] + "_"
        numerator = self.kwargs["numerator"]
        result_dict = {prefix + "inverse_max": np.zeros(len(X))}

        base_dir = Path("input/atma5/spectrum")
        for i, row in progress_bar(X.iterrows(), total=len(X)):
            spectrum = pd.read_csv(
                base_dir / row.spectrum_filename, sep="\t", header=None)
            result_dict[prefix +
                        "inverse_max"][i] = numerator / spectrum[1].max()
        df = pd.DataFrame(result_dict)
        df.columns = [prefix + c for c in df.columns]
        return df


class SpecMaxThreshed:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_transform(self, X: pd.DataFrame):
        return self.transform(X)

    def transform(self, X: pd.DataFrame):
        threshold = self.kwargs["threshold"]
        result_dict = {"max_threshed": np.zeros(len(X))}

        base_dir = Path("input/atma5/spectrum")
        for i, row in progress_bar(X.iterrows(), total=len(X)):
            spectrum = pd.read_csv(
                base_dir / row.spectrum_filename, sep="\t", header=None)
            spec_max = spectrum[1].max()
            result_dict["max_threshed"][
                i] = spec_max if spec_max <= threshold else threshold

        df = pd.DataFrame(result_dict)
        return df


class SpecStdSkewKurt:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_transform(self, X: pd.DataFrame):
        return self.transform(X)

    def transform(self, X: pd.DataFrame):
        operations = ["std", "skew", "kurt"]
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


class SpecDistance:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_transform(self, X: pd.DataFrame):
        return self.transform(X)

    def transform(self, X: pd.DataFrame):
        agg = X.groupby("spectrum_filename")["intensity"].agg(
            ["min", "max", "mean", "median"])
        features = pd.DataFrame(
            index=agg.index,
            columns=[
                "spectrum_max_min", "spectrum_max_mean", "spectrum_max_median"
            ])
        features["spectrum_max_min"] = agg["max"] - agg["min"]
        features["spectrum_max_mean"] = agg["max"] - agg["mean"]
        features["spectrum_max_median"] = agg["max"] - agg["median"]

        return features.reset_index(drop=True)


class WindowedMeanMax:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_transform(self, X: pd.DataFrame):
        return self.transform(X)

    def transform(self, X: pd.DataFrame):
        unique_filenames = X["spectrum_filename"].unique()
        features = {}

        window_sizes = self.kwargs["window_size"]
        for ws in window_sizes:
            features[f"windowed_mean_max_ws_{ws}"] = np.zeros(
                len(unique_filenames))

        for i, filename in enumerate(progress_bar(unique_filenames)):
            windowed_means: Dict[int, list] = {ws: [] for ws in window_sizes}
            spec = X.query(f"spectrum_filename == '{filename}'")
            x = spec["intensity"].values

            for ws in window_sizes:
                for j in range(len(spec) - ws + 1):
                    windowed_means[ws].append(x[j:j + ws].mean())

                features[f"windowed_mean_max_ws_{ws}"][i] = np.max(
                    windowed_means[ws])
        return pd.DataFrame(features)


class ReshapeAndScale:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.overall_max = 0.0
        self.overall_min = 100000000
        self.overall_mean = 0.0
        self.overall_std = 0.0

    def fit_transform(self, X: pd.DataFrame):
        if self.kwargs["scaling"] == "min_max":
            self.overall_max = X["intensity"].max()
            self.overall_min = X["intensity"].min()
        elif self.kwargs["scaling"] == "normalize":
            self.overall_mean = X["intensity"].mean()
            self.overall_std = X["intensity"].std()
        return self.transform(X)

    def transform(self, X: pd.DataFrame):
        unique_filenames = X["spectrum_filename"].unique()
        new_array = np.zeros((len(unique_filenames), 511))

        head = 0
        spectrum_filenames = X["spectrum_filename"].values
        intensity = X["intensity"].values

        for i, filename in enumerate(progress_bar(unique_filenames)):
            len_f = (spectrum_filenames == filename).sum()
            x = intensity[head:head + 511]

            if self.kwargs["scaling"] == "min_max":
                x = x / (self.overall_max - self.overall_min)
            elif self.kwargs["scaling"] == "normalize":
                x = (x - self.overall_mean) / self.overall_std
            new_array[i] = x
            head = head + len_f
        return pd.DataFrame(new_array, columns=list(range(511)))


class FittingCauchy:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_transform(self, X: pd.DataFrame):
        return self.transform(X)

    def transform(self, X: pd.DataFrame):
        X["peak_distance"] = (X["params2"] - X["params5"]).abs()
        X["cavity_ratio"] = X["params1"] / X["params4"]
        X["peak_area_ratio"] = X["params1"] / X["params3"]

        return X[["peak_distance", "cavity_ratio", "peak_area_ratio"]]
