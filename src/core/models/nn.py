import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data

import src.utils as utils

from typing import List, Tuple, Optional, Union
from pathlib import Path

from catalyst.core import Callback, CallbackOrder, State
from catalyst.dl import SupervisedRunner
from catalyst.utils import get_device
from scipy.stats import cauchy
from sklearn.metrics import average_precision_score, roc_auc_score

from .base import NNModel, Matrix


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def get_criterion(criterion_params: dict):
    name = criterion_params["name"]
    params = {} if criterion_params.get(
        "params") is None else criterion_params["params"]
    if name == "FocalLoss":
        return FocalLoss(**params)
    else:
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


class FileDataset(data.Dataset):
    def __init__(self,
                 df: Matrix,
                 target: Optional[Matrix],
                 file_dir: str,
                 scale="normalize"):
        self.values = df
        self.target = target
        self.file_dir = Path(file_dir)
        self.scale = scale

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx: int):
        filename = self.values[idx][0]
        df = pd.read_csv(self.file_dir / filename, sep="\t", header=None)
        spectrum = df[1].values
        if self.scale == "normalize":
            spectrum = (spectrum - spectrum.mean()) / spectrum.std()
        else:
            spectrum = (spectrum - spectrum.min()) / (
                spectrum.max() - spectrum.min())
        spectrum = spectrum[:511].astype(np.float32)
        if self.target is not None:
            return spectrum, self.target[idx]
        else:
            return spectrum


class RawFittingDataset(data.Dataset):
    def __init__(self,
                 df: Matrix,
                 target: Optional[Matrix],
                 file_dir: str,
                 fitting_file_dir: str,
                 scale="normalize",
                 crop=False,
                 flip=False,
                 noise=False,
                 peak=False):
        self.values = df
        self.target = target
        self.file_dir = Path(file_dir)
        self.fitting_file_dir = Path(fitting_file_dir)
        self.scale = scale

        self.crop = crop
        self.flip = flip
        self.noise = noise
        self.peak = peak

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx: int):
        filename = self.values[idx][0]
        df = pd.read_csv(self.file_dir / filename, sep="\t", header=None)
        fitting = pd.read_csv(
            self.fitting_file_dir / filename, sep="\t", header=None)

        spectrum = df[1].values
        spectrum_fitting = fitting[1].values
        if self.noise:
            scale = np.random.randint(50, 200)
            noise = scale * np.random.normal(len(spectrum))
            spectrum = spectrum + noise

        if self.peak:
            idxmax = fitting[0].values.argmax()
            sign = 1 if np.random.rand() > 0.5 else -1
            peak_pos = fitting[0].values[idxmax] + np.random.randint(
                50, 150) * sign
            scale = np.abs(np.random.normal() * 40)

            false_peak = cauchy.pdf(df[0].values, peak_pos, scale)
            ratio = df[1].max() / false_peak.max()

            false_peak = false_peak * ratio * min(np.random.rand(), 0.8)

            spectrum = spectrum + false_peak

        if self.crop:
            start = np.random.randint(0, 111)
            spectrum = spectrum[start:start + 400].astype(np.float32)
            spectrum_fitting = spectrum_fitting[start:start + 400].astype(
                np.float32)
        else:
            spectrum = spectrum[:511].astype(np.float32)
            spectrum_fitting = spectrum_fitting[:511].astype(np.float32)

        if self.scale == "normalize":
            spectrum = (spectrum - spectrum.mean()) / spectrum.std()
            spectrum_fitting = (spectrum_fitting - spectrum_fitting.mean()
                                ) / spectrum_fitting.std()
        else:
            spectrum = (spectrum - spectrum.min()) / (
                spectrum.max() - spectrum.min())
            spectrum_fitting = (
                spectrum_fitting - spectrum_fitting.min() /
                (spectrum_fitting.max() - spectrum_fitting.min()))

        if self.flip:
            if np.random.rand() > 0.5:
                spectrum = np.flip(spectrum).copy()
                spectrum_fitting = np.flip(spectrum_fitting).copy()

        x = np.asarray([spectrum, spectrum_fitting]).astype(np.float32)
        if self.target is not None:
            return x, self.target[idx]
        else:
            return x


class FittingDataset(data.Dataset):
    def __init__(self,
                 df: Matrix,
                 target: Optional[Matrix],
                 file_dir: str,
                 scale="normalize",
                 crop=False,
                 flip=False,
                 noise=False,
                 peak=False):
        self.values = df
        self.target = target
        self.file_dir = Path(file_dir)
        self.scale = scale

        self.crop = crop
        self.flip = flip
        self.noise = noise
        self.peak = peak

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx: int):
        filename = self.values[idx][0]
        params2 = self.values[idx][1]
        df = pd.read_csv(self.file_dir / filename, sep="\t", header=None)

        spectrum = df[1].values
        if self.noise:
            scale = np.random.randint(50, 200)
            noise = scale * np.random.normal(len(spectrum))
            spectrum = spectrum + noise

        if self.peak:
            sign = 1 if np.random.rand() > 0.5 else -1
            peak_pos = params2 + np.random.randint(50, 150) * sign
            scale = np.abs(np.random.normal() * 40)

            false_peak = cauchy.pdf(df[0].values, peak_pos, scale)
            ratio = df[1].max() / false_peak.max()

            false_peak = false_peak * ratio * min(np.random.rand(), 0.8)

            spectrum = spectrum + false_peak

        if self.crop:
            start = np.random.randint(0, 111)
            spectrum = spectrum[start:start + 400].astype(np.float32)
        else:
            spectrum = spectrum[:511].astype(np.float32)

        if self.scale == "normalize":
            spectrum = (spectrum - spectrum.mean()) / spectrum.std()
        else:
            spectrum = (spectrum - spectrum.min()) / (
                spectrum.max() - spectrum.min())

        if self.flip:
            if np.random.rand() > 0.5:
                spectrum = np.flip(spectrum).copy()
        if self.target is not None:
            return spectrum, self.target[idx]
        else:
            return spectrum


def get_loader(loader_params: dict, df: Matrix, target: Optional[Matrix]):
    dataset_type = loader_params.get("dataset_type")
    if dataset_type == "from_file":
        scale = "normalize" if loader_params.get(
            "scale") is None else "min_max"
        dataset = FileDataset(df, target, loader_params["file_dir"], scale)
        params = loader_params.copy()
        params.pop("dataset_type")
        params.pop("file_dir")
        if scale is not None:
            params.pop("scale")
    elif dataset_type == "with_fitting":
        scale = "normalize" if loader_params.get(
            "scale") is None else "min_max"
        crop = loader_params["crop"]
        flip = loader_params["flip"]
        noise = loader_params["noise"]
        peak = loader_params.get("peak")
        if peak is None:
            peak = False
        dataset = FittingDataset(  # type: ignore
            df,
            target,
            loader_params["file_dir"],
            scale,
            crop=crop,
            flip=flip,
            noise=noise,
            peak=peak)
        params = loader_params.copy()
        params.pop("dataset_type")
        params.pop("file_dir")
        if scale is not None:
            params.pop("scale")
        params.pop("crop")
        params.pop("flip")
        params.pop("noise")
        if params.get("peak") is not None:
            params.pop("peak")
    elif dataset_type == "raw_and_fitting":
        scale = "normalize" if loader_params.get(
            "scale") is None else "min_max"
        crop = loader_params["crop"]
        flip = loader_params["flip"]
        noise = loader_params["noise"]
        dataset = RawFittingDataset(  # type: ignore
            df,
            target,
            loader_params["file_dir"],
            loader_params["fitting_file_dir"],
            scale,
            crop=crop,
            flip=flip,
            noise=noise)
        params = loader_params.copy()
        params.pop("dataset_type")
        params.pop("file_dir")
        params.pop("fitting_file_dir")
        if scale is not None:
            params.pop("scale")
        params.pop("crop")
        params.pop("flip")
        params.pop("noise")
    else:
        dataset = TabularDataset(df, target)  # type: ignore
    return data.DataLoader(dataset, **params)


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
        if x.ndim == 2:
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


class RandomCrop:
    def __init__(self, cropped_size: int = 500):
        self.cropped_size = cropped_size

    def __call__(self, x: np.ndarray):
        original_size = len(x)
        start = np.random.randint(0, original_size - self.cropped_size)
        return x[start:start + self.cropped_size]


class Flip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x: np.ndarray):
        if np.random.rand() <= self.p:
            return x[::-1]
        else:
            return x


class Conv1DModel(NNModel):
    def __init__(self, mode: str, log_dir: Union[str, Path]):
        super().__init__(mode)

        self.log_dir = Path(log_dir) if isinstance(log_dir, str) else log_dir
        self.train_params: Optional[dict] = None

    def fit(self, X_train: Matrix, Y_train: Matrix,
            valid_sets: List[Tuple[Matrix, Matrix]],
            valid_names: Optional[List[str]], model_params: dict,
            train_params: dict):
        data_loaders = {}
        self.train_params = train_params

        utils.set_seed(train_params["seed"])

        loader_params = train_params["loader"]

        data_loaders["train"] = get_loader(loader_params["train"], X_train,
                                           Y_train.astype(float))
        for i, (X, y) in enumerate(valid_sets):
            if valid_names is None:
                if i == 0:
                    name = "valid"
                else:
                    name = f"valid_{i}"
                data_loaders[name] = get_loader(loader_params["valid"], X,
                                                y.astype(float))
            else:
                name = valid_names[i]
                data_loaders[name] = get_loader(loader_params["valid"], X,
                                                y.astype(float))

        callback_params = train_params["callback"]
        callbacks = get_callbacks(callback_params)

        criterion_params = train_params["criterion"]
        criterion = get_criterion(criterion_params)
        model = CNN1D(model_params)

        optimizer_params = train_params["optimizer"]
        optimizer = get_optimizer(model, optimizer_params)

        scheduler_params = train_params["scheduler"]
        scheduler = get_scheduler(optimizer, scheduler_params)

        if train_params.get("main_metric") is not None:
            main_metric = train_params["main_metric"]
            minimize_metric = train_params["minimize_metric"]
        else:
            main_metric = "loss"
            minimize_metric = True

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
            callbacks=callbacks,
            main_metric=main_metric,
            minimize_metric=minimize_metric)

        self.model = model  # type: ignore

    def predict(self, X_test: Matrix, tta: int = 1):
        self._assert_if_untrained()
        weights_path = self.log_dir / "checkpoints/best_full.pth"
        weights = torch.load(weights_path)
        self.model.load_state_dict(weights["model_state_dict"])  # type: ignore
        self.model.to(get_device())  # type: ignore
        self.model.eval()  # type: ignore
        loader_params = self.train_params["loader"]["valid"]  # type: ignore
        loader = get_loader(loader_params, X_test, None)

        batch_size = loader_params["batch_size"]

        predictions = np.zeros(len(X_test))
        device = get_device()

        tta_preds = []
        for seed in range(tta):
            utils.set_seed(seed)
            for i, x_batch in enumerate(loader):
                with torch.no_grad():
                    x_batch = x_batch.to(device)
                    preds = self.model(  # type: ignore
                        x_batch).detach().cpu().numpy()

                predictions[i * batch_size:(i + 1) *
                            batch_size] = preds.reshape(-1)
            tta_preds.append(predictions)

        tta_prediction = np.asarray(tta_preds).mean(axis=0)
        return tta_prediction

    def _assert_if_untrained(self):
        if self.model is None:
            msg = "Model has not been trained yet."
            msg += "Call `.fit()` method first."
            raise AttributeError(msg)
