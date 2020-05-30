import numpy as np
import pandas as pd

import src.core.preprocess.functional as F

from typing import Optional, Union

from src.core.states import RunningState

from .base import Callback, CallbackOrder


# on_preprocess_start
class TrainTestSameColumnsCallback(Callback):
    signature = "preprocess"
    callback_order = CallbackOrder.ASSERTION

    def on_preprocess_start(self, state: RunningState):
        train_features = state.features["main"]["train"]
        test_features = state.features["main"]["test"]

        train_columns = set(train_features.columns)
        test_columns = set(test_features.columns)

        in_train_not_in_test = train_columns - test_columns
        in_test_not_in_train = test_columns - train_columns

        if len(in_train_not_in_test) > 0 or len(in_test_not_in_train) > 0:
            msg = "Find column inconsistency between train and test."
            msg += "Train has these columns, but test doesn't: [\n"
            for column in in_train_not_in_test:
                msg += "    " + column + "\n"
            msg += "]\n Test has these columns, but train doesn't: [\n"
            for column in in_test_not_in_train:
                msg += "    " + column + "\n"
            msg += "]. aborting."
            raise AssertionError(msg)


class RemoveConstantCallback(Callback):
    signature = "preprocess"
    callback_order = CallbackOrder.LOWEST

    def on_preprocess_start(self, state: RunningState):
        train_features = state.features["main"]["train"]
        test_features = state.features["main"]["test"]

        to_remove_train = F.find_constant_columns(train_features)
        to_remove_test = F.find_constant_columns(test_features)

        to_remove = list(set(to_remove_train).union(to_remove_test))

        if len(to_remove) > 0:
            state.features["main"]["train"] = train_features.drop(
                to_remove, axis=1)
            state.features["main"]["test"] = test_features.drop(
                to_remove, axis=1)

            msg = "Found constant columns: [\n"
            for column in to_remove:
                msg += "    " + column + "\n"
            msg += "]."
            state.logger.info(msg)


class RemoveDuplicatedCallback(Callback):
    signature = "preprocess"
    callback_order = CallbackOrder.LOWEST

    def on_preprocess_start(self, state: RunningState):
        train_features = state.features["main"]["train"]
        test_features = state.features["main"]["test"]

        to_remove_train = F.find_duplicated_columns(train_features)
        to_remove_test = F.find_duplicated_columns(test_features)

        to_remove = list(set(to_remove_train).union(to_remove_test))

        if len(to_remove) > 0:
            state.features["main"]["train"] = train_features.drop(
                to_remove, axis=1)
            state.features["main"]["test"] = test_features.drop(
                to_remove, axis=1)

            msg = "Found duplicated columns: [\n"
            for column in to_remove:
                msg += "    " + column + "\n"
            msg += "]."
            state.logger.info(msg)


class RemoveCorrelatedCallback(Callback):
    signature = "preprocess"
    callback_order = CallbackOrder.LOWEST

    def __init__(self, threshold=0.995):
        self.threshold = threshold

    def on_preprocess_start(self, state: RunningState):
        train_features = state.features["main"]["train"]
        test_features = state.features["main"]["test"]

        to_remove_train = F.find_correlated_columns(
            train_features, threshold=self.threshold)
        to_remove_test = F.find_correlated_columns(
            test_features, threshold=self.threshold)

        to_remove = list(set(to_remove_train).union(to_remove_test))

        if len(to_remove) > 0:
            state.features["main"]["train"] = train_features.drop(
                to_remove, axis=1)
            state.features["main"]["test"] = test_features.drop(
                to_remove, axis=1)

            msg = "Found highly correlated columns: [\n"
            for column in to_remove:
                msg += "    " + column + "\n"
            msg += "]."
            state.logger.info(msg)


class RemoveBasedOnImportanceCallback(Callback):
    signature = "preprocess"
    callback_order = CallbackOrder.ASSERTION

    def __init__(self,
                 n_remains: Optional[Union[int, float]],
                 n_delete: Optional[Union[int, float]] = None,
                 importance_key="av",
                 remove_low=False):
        if n_remains is None and n_delete is None:
            raise ValueError("Either `n_remains` or `n_delete` must be given.")

        if n_remains is not None and n_delete is not None:
            raise ValueError("Both `n_remains` and `n_delete` are given,"
                             "choose either of them to use.")

        self.n_remains = n_remains
        self.n_delete = n_delete
        self.importance_key = importance_key
        self.remove_low = remove_low

    def on_preprocess_start(self, state: RunningState):
        train_features = state.features["main"]["train"]
        test_features = state.features["main"]["test"]

        importance = state.importances.get(self.importance_key)
        if importance is None:
            msg = f"Importance dict/DataFrame {self.importance_key} doesn't exist. Skipping."
            state.logger.info(msg)
            return

        if isinstance(importance, dict):
            keys, values = list(importance.keys()), list(importance.values())
            columns = list(np.array(keys)[np.argsort(values)])
        elif isinstance(importance, pd.DataFrame):
            if "value" not in importance.columns or "feature" not in importance.columns:
                raise KeyError(
                    "Feature-importance DataFrame must have `importance` and `feature` columns"
                )
            columns = importance.sort_values(by="value")["feature"].tolist()

        missing_in_train = set(columns) - set(train_features.columns)
        missing_in_test = set(columns) - set(test_features.columns)

        if len(missing_in_train) > 0:
            msg = "Found features that are present in importances"
            msg += " but not in train DataFrame or test DataFrame.\n"
            msg += "Features missing in train: [\n"
            for column in missing_in_train:
                msg += "    " + column + "\n"
            msg += "]. Features missing in test: [\n"
            for column in missing_in_test:
                msg += "    " + column + "\n"
            msg += "]. Aborting."
            raise KeyError(msg)

        if self.n_remains is not None:
            if isinstance(self.n_remains, int):
                n_delete = len(columns) - self.n_remains
            elif isinstance(self.n_remains, float):
                n_remains = int(len(columns) * self.n_remains)
                n_delete = len(columns) - n_remains
        else:
            if isinstance(self.n_delete, int):
                n_delete = self.n_delete
            elif isinstance(self.n_delete, float):
                n_delete = int(len(columns) * self.n_delete)

        if self.remove_low:
            to_remove = columns[:n_delete]
        else:
            to_remove = reversed(columns)[:n_delete]  # type: ignore

        if len(to_remove) > 0:
            state.features["main"]["train"] = train_features.drop(
                to_remove, axis=1)
            state.features["main"]["test"] = test_features.drop(
                to_remove, axis=1)

            msg = f"Remove {'low' if self.remove_low else 'high'}"
            msg += f" {self.importance_key} importance columns: [\n"
            for column in to_remove:
                msg += "    " + column + "\n"
            msg += "]."
            state.logger.info(msg)
