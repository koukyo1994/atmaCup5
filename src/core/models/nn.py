import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data

from typing import List, Tuple, Optional, Union
from pathlib import Path

from catalyst.core import Callback, CallbackOrder, State
from catalyst.dl import SupervisedRunner
from catalyst.utils import get_device
from sklearn.metrics import average_precision_score, roc_auc_score

from .base import NNModel, Matrix


def get_criterion(criterion_params: dict):
    name = criterion_params["name"]
    params = {} if criterion_params.get(
        "params") is None else criterion_params["params"]
    return nn.__getattribute__(name)(**params)


def get_optimizer(model, optimizer_params: dict):
    name = optimizer_params["name"]
    return optim.__getattribute__(name)(model.parameters(),
                                        **optimizer_params["params"])


def get_scheduler(optimizer, scheduler_params: dict):
    name = scheduler_params["name"]
    return lr_scheduler.__getattribute__(name)(optimizer,
                                               **scheduler_params["params"])


class TabularDataset(data.Dataset):
    def __init__(self, df: Matrix, target: Optional[Matrix]):
        self.values = df
        self.target = target

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx: int):
        x = self.values[idx].astype(np.float32)
        if self.target is not None:
            return x, self.target[idx]
        else:
            return x


def get_loader(loader_params: dict, df: Matrix, target: Optional[Matrix]):
    dataset = TabularDataset(df, target)
    return data.DataLoader(dataset, **loader_params)


class Conv1dBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int):
        super(Conv1dBNReLU, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride), nn.BatchNorm1d(out_channels), nn.ReLU())

    def forward(self, x):
        return self.seq(x)


class SpatialAttention1d(nn.Module):
    def __init__(self, in_channels: int):
        super(SpatialAttention1d, self).__init__()
        self.squeeze = nn.Conv1d(in_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z


class GAB1d(nn.Module):
    def __init__(self, in_channels: int, reduction=4):
        super(GAB1d, self).__init__()
        self.global_avgpool = nn.AdaptiveMaxPool1d(1)
        self.conv1 = nn.Conv1d(
            in_channels, in_channels // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(
            in_channels // reduction, in_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z


class SCse1d(nn.Module):
    def __init__(self, in_channels: int):
        super(SCse1d, self).__init__()
        self.satt = SpatialAttention1d(in_channels)
        self.catt = GAB1d(in_channels)

    def forward(self, x):
        return self.satt(x) + self.catt(x)


class CNN1D(nn.Module):
    def __init__(self, model_params: dict):
        super(CNN1D, self).__init__()

        modules = []
        architecture = model_params['architecture']
        for module in architecture:
            name = module["name"]
            params = {} if module.get("params") is None else module["params"]
            if module["type"] == "torch":
                modules.append(nn.__getattribute__(name)(**params))
            else:
                modules.append(globals().get(name)(**params))  # type: ignore
        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1)
        return self.seq(x).view(batch_size)


class mAPCallback(Callback):
    def __init__(self, prefix: str = "mAP"):
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix

    def on_loader_start(self, state: State):
        self.prediction: List[np.ndarray] = []
        self.target: List[np.ndarray] = []

    def on_batch_end(self, state: State):
        targ = state.input["targets"].detach().cpu().numpy()
        out = state.output["logits"].detach().cpu().numpy()

        self.prediction.append(out)
        self.target.append(targ)

        score = average_precision_score(targ, out)
        score = np.nan_to_num(score)
        state.batch_metrics[self.prefix] = score

    def on_loader__end(self, state: State):
        y_pred = np.concatenate(self.prediction, axis=0)
        y_true = np.concatenate(self.target, axis=0)
        score = average_precision_score(y_true, y_pred)
        state.epoch_metrics[self.prefix] = score


class AUCCallback(Callback):
    def __init__(self, prefix: str = "auc"):
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix

    def on_loader_start(self, state: State):
        self.prediction: List[np.ndarray] = []
        self.target: List[np.ndarray] = []

    def on_batch_end(self, state: State):
        targ = state.input["targets"].detach().cpu().numpy()
        out = state.output["logits"].detach().cpu().numpy()

        self.prediction.append(out)
        self.target.append(targ)

        score = roc_auc_score(targ, out)
        score = np.nan_to_num(score)
        state.batch_metrics[self.prefix] = score

    def on_loader__end(self, state: State):
        y_pred = np.concatenate(self.prediction, axis=0)
        y_true = np.concatenate(self.target, axis=0)
        score = roc_auc_score(y_true, y_pred)
        state.epoch_metrics[self.prefix] = score


def get_callbacks(callback_params: List[dict]):
    callbacks = []
    for params in callback_params:
        name = params["name"]
        callback_param = params["params"]
        callbacks.append(globals().get(name)(**callback_param))  # type: ignore
    return callbacks


class Conv1DModel(NNModel):
    def __init__(self, mode: str, log_dir: Union[str, Path]):
        super().__init__(mode)

        self.log_dir = Path(log_dir) if isinstance(log_dir, str) else log_dir

    def fit(self, X_train: Matrix, Y_train: Matrix,
            valid_sets: List[Tuple[Matrix, Matrix]],
            valid_names: Optional[List[str]], model_params: dict,
            train_params: dict):
        data_loaders = {}

        loader_params = train_params["loader"]

        data_loaders["train"] = get_loader(loader_params["train"], X_train,
                                           Y_train)
        for i, (X, y) in enumerate(valid_sets):
            if valid_names is None:
                if i == 0:
                    name = "valid"
                else:
                    name = f"valid_{i}"
                data_loaders[name] = get_loader(loader_params["valid"], X, y)
            else:
                name = valid_names[i]
                data_loaders[name] = get_loader(loader_params["valid"], X, y)

        callback_params = train_params["callback"]
        callbacks = get_callbacks(callback_params)

        criterion_params = train_params["criterion"]
        criterion = get_criterion(criterion_params)
        model = CNN1D(model_params)

        optimizer_params = train_params["optimizer"]
        optimizer = get_optimizer(model, optimizer_params)

        scheduler_params = train_params["scheduler"]
        scheduler = get_scheduler(optimizer, scheduler_params)

        runner = SupervisedRunner(device=get_device())
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=data_loaders,
            scheduler=scheduler,
            logdir=self.log_dir,
            verbose=True,
            num_epochs=train_params["num_epochs"],
            callbacks=callbacks)

        self.model = model  # type: ignore

    def predict(self, X_test: Matrix):
        self._assert_if_untrained()
        weights_path = self.log_dir / "checkpoints/best_full.pth"
        weights = torch.load(weights_path)
        self.model.load_state_dict(weights["model_state_dict"])  # type: ignore
        self.model.to(get_device())  # type: ignore
        self.model.eval()  # type: ignore
        loader_params = {"batch_size": 512, "shuffle": False, "num_workers": 4}
        loader = get_loader(loader_params, X_test, None)

        predictions = np.zeros(len(X_test))
        device = get_device()
        for i, x_batch in enumerate(loader):
            with torch.no_grad():
                x_batch = x_batch.to(device)
                preds = self.model(  # type: ignore
                    x_batch).detach().cpu().numpy()

            predictions[i * 512:(i + 1) * 512] = preds.reshape(-1)
        return predictions

    def _assert_if_untrained(self):
        if self.model is None:
            msg = "Model has not been trained yet."
            msg += "Call `.fit()` method first."
            raise AttributeError(msg)
