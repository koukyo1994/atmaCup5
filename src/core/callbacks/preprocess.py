import src.core.preprocess.functional as F

from src.core.states import RunningState

from .base import Callback, CallbackOrder


# on_preprocess_start
class RemoveConstantCallback(Callback):
    callback_order = CallbackOrder.LOWEST

    def on_preprocess_start(self, state: RunningState):
        train_features = state.features["train"]
        test_features = state.features["test"]

        to_remove_train = F.find_constant_columns(train_features)
        to_remove_test = F.find_constant_columns(test_features)

        to_remove = list(set(to_remove_train).union(to_remove_test))

        state.features["train"] = train_features.drop(to_remove, axis=1)
        state.features["test"] = test_features.drop(to_remove, axis=1)


class RemoveDuplicatedCallback(Callback):
    callback_order = CallbackOrder.LOWEST

    def on_preprocess_start(self, state: RunningState):
        train_features = state.features["train"]
        test_features = state.features["test"]

        to_remove_train = F.find_duplicated_columns(train_features)
        to_remove_test = F.find_duplicated_columns(test_features)

        to_remove = list(set(to_remove_train).union(to_remove_test))

        state.features["train"] = train_features.drop(to_remove, axis=1)
        state.features["test"] = test_features.drop(to_remove, axis=1)


class RemoveCorrelatedCallback(Callback):
    callback_order = CallbackOrder.LOWEST

    def __init__(self, threshold=0.995):
        self.threshold = threshold

    def on_preprocess_start(self, state: RunningState):
        train_features = state.features["train"]
        test_features = state.features["test"]

        to_remove_train = F.find_correlated_columns(
            train_features, threshold=self.threshold)
        to_remove_test = F.find_correlated_columns(
            test_features, threshold=self.threshold)

        to_remove = list(set(to_remove_train).union(to_remove_test))

        state.features["train"] = train_features.drop(to_remove, axis=1)
        state.features["test"] = test_features.drop(to_remove, axis=1)
