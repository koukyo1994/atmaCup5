import logging

from .lgbm import LGBModel, log_evaluation


def get_model(config: dict, logger: logging.Logger):
    if config.get("name") == "lgbm":
        feval = None
        callbacks = []
        if config.get("feval") is not None:
            feval = globals().get(config.get("feval"))  # type: ignore

        if config.get("callbacks") is not None:
            callback_configs = config["callbacks"]
            for callback_conf in callback_configs:
                callback_name = callback_conf.get("name")
                callback_params = callback_conf.get("params")
                if "logger" in callback_params.keys():
                    callback_params["logger"] = logger
                callbacks.append(globals().get(callback_name)(  # type: ignore
                    **callback_params))
        return LGBModel(mode="", callbacks=callbacks, feval=feval)
    else:
        raise NotImplementedError
