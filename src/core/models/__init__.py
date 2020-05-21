from .lgbm import LGBModel


def get_model(config: dict):
    if config.get("name") == "lgbm":
        return LGBModel
    else:
        raise NotImplementedError
