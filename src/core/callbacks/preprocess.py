import src.core.preprocess.functional as F

from src.core.states import RunningState

from .base import Callback, CallbackOrder


# on_preprocess_start
class TrainTestSameColumnsCallback(Callback):
    callback_order = CallbackOrder.ASSERTION

    def on_preprocess_start(self, state: RunningState):
        train_features = state.features["train"]
        test_features = state.features["test"]

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
    callback_order = CallbackOrder.LOWEST

    def on_preprocess_start(self, state: RunningState):
        train_features = state.features["train"]
        test_features = state.features["test"]

        to_remove_train = F.find_constant_columns(train_features)
        to_remove_test = F.find_constant_columns(test_features)

        to_remove = list(set(to_remove_train).union(to_remove_test))

        if len(to_remove) > 0:
            state.features["train"] = train_features.drop(to_remove, axis=1)
            state.features["test"] = test_features.drop(to_remove, axis=1)

            msg = "Found constant columns: [\n"
            for column in to_remove:
                msg += "    " + column + "\n"
            msg += "]."
            state.logger.info(msg)


class RemoveDuplicatedCallback(Callback):
    callback_order = CallbackOrder.LOWEST

    def on_preprocess_start(self, state: RunningState):
        train_features = state.features["train"]
        test_features = state.features["test"]

        to_remove_train = F.find_duplicated_columns(train_features)
        to_remove_test = F.find_duplicated_columns(test_features)

        to_remove = list(set(to_remove_train).union(to_remove_test))

        if len(to_remove) > 0:
            state.features["train"] = train_features.drop(to_remove, axis=1)
            state.features["test"] = test_features.drop(to_remove, axis=1)

            msg = "Found duplicated columns: [\n"
            for column in to_remove:
                msg += "    " + column + "\n"
            msg += "]."
            state.logger.info(msg)


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

        if len(to_remove) > 0:
            state.features["train"] = train_features.drop(to_remove, axis=1)
            state.features["test"] = test_features.drop(to_remove, axis=1)

            msg = "Found highly correlated columns: [\n"
            for column in to_remove:
                msg += "    " + column + "\n"
            msg += "]."
            state.logger.info(msg)
