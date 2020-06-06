from .lgbm import LGBModel, log_evaluation, pr_auc
from .nn import Conv1DModel

from src.core.states import RunningState


def get_model(config: dict, state: RunningState):
    if config.get("name") == "lgbm":
        feval = None
        callbacks = []
        if config.get("feval") is not None:
            feval = globals().get(config.get("feval"))  # type: ignore

        if config.get("callbacks") is not None:
            callback_configs = config["callbacks"]
            for callback_conf in callback_configs:
                if len(callback_conf) == 0:
                    continue
                callback_name = callback_conf.get("name")
                callback_params = callback_conf.get("params")
                if "logger" in callback_params.keys():
                    callback_params["logger"] = state.logger
                callbacks.append(globals().get(callback_name)(  # type: ignore
                    **callback_params))
        return LGBModel(mode="", callbacks=callbacks, feval=feval)
    elif config.get("name") == "conv1d":
        fold_signature = f"fold{state.misc['current_fold_id'] + 1}"
        return Conv1DModel(
            mode="",
            log_dir=state.output_dir / state.config["identifier"] /
            fold_signature)
    else:
        raise NotImplementedError
