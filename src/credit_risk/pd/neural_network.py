"""Arquitectura y envoltorio de entrenamiento del MLP de clasificación PD."""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def seleccionar_dispositivo() -> torch.device:
    """Usa CUDA, MPS o CPU según esté disponible."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fijar_semilla(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def matriz_densa_float32(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


class PDNet(nn.Module):
    """MLP binario que produce logits para ``BCEWithLogitsLoss``."""

    def __init__(self, input_dim: int, config: dict):
        super().__init__()
        activations = {"relu": nn.ReLU, "tanh": nn.Tanh}
        try:
            activation = activations[str(config["activation"]).lower()]
        except KeyError as exc:
            raise ValueError(f"Activación no reconocida: {config['activation']}") from exc

        blocks: list[nn.Module] = []
        previous = int(input_dim)
        for _ in range(int(config["n_layers"])):
            blocks.extend([nn.Linear(previous, int(config["width"])), activation()])
            if float(config["dropout"]) > 0:
                blocks.append(nn.Dropout(float(config["dropout"])))
            previous = int(config["width"])
        self.hidden = nn.Sequential(*blocks)
        self.output = nn.Linear(previous, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(values)).squeeze(-1)


class PDNeuralNetwork:
    """Entrena, evalúa y serializa el MLP con parada temprana en validación."""

    def __init__(self, config: dict, seed: int, device=None):
        self.config = dict(config)
        self.seed = int(seed)
        self.device = torch.device(device) if device is not None else seleccionar_dispositivo()
        self.model: PDNet | None = None
        self.input_dim: int | None = None
        self.best_epoch = 0
        self.epochs_run = 0
        self.validation_bce = math.inf
        self.fit_time_sec = 0.0
        self.train_history: list[dict[str, float]] = []
        self.classes_ = np.asarray([0, 1])

    @property
    def n_features_in_(self):
        """Dimensión transformada para el contrato de inferencia de la app."""
        return self.input_dim

    @property
    def n_parameters(self) -> int:
        if self.model is None:
            return 0
        return sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)

    def _penalty(self) -> torch.Tensor:
        weights = [parameter for name, parameter in self.model.named_parameters() if "weight" in name]
        value = torch.zeros((), device=self.device)
        if self.config["reg_type"] in {"l1", "elastic_net"}:
            value = value + float(self.config["l1"]) * sum(weight.abs().sum() for weight in weights)
        if self.config["reg_type"] in {"l2", "elastic_net"}:
            value = value + float(self.config["l2"]) * sum((weight**2).sum() for weight in weights) / 2
        return value

    def _make_optimizer(self):
        name = self.config["optimizer"]
        lr = float(self.config["learning_rate"])
        if name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        if name == "nadam":
            return torch.optim.NAdam(self.model.parameters(), lr=lr)
        if name == "sgd_momentum":
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        raise ValueError(f"Optimizador no reconocido: {name}")

    def fit(self, X_train, y_train, X_val, y_val, max_epochs: int = 100,
            patience: int = 8, verbose: bool = False):
        """Ajusta con train y conserva el estado con menor BCE en validación."""
        fijar_semilla(self.seed)
        X_train = matriz_densa_float32(X_train)
        X_val = matriz_densa_float32(X_val)
        y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
        y_val = np.asarray(y_val, dtype=np.float32).reshape(-1)
        self.input_dim = X_train.shape[1]
        self.model = PDNet(self.input_dim, self.config).to(self.device)
        optimizer = self._make_optimizer()
        loss_function = nn.BCEWithLogitsLoss()
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=int(self.config["batch_size"]),
            shuffle=True,
            generator=generator,
        )
        X_val_tensor = torch.from_numpy(X_val).to(self.device)
        y_val_tensor = torch.from_numpy(y_val).to(self.device)
        best_loss = math.inf
        best_state = None
        stale_epochs = 0
        self.train_history = []
        started = time.perf_counter()

        for epoch in range(1, int(max_epochs) + 1):
            self.model.train()
            total_train_bce = 0.0
            observations = 0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                bce = loss_function(self.model(X_batch), y_batch)
                objective = bce + self._penalty()
                if not torch.isfinite(objective):
                    raise FloatingPointError(f"Pérdida no finita en época {epoch}, semilla {self.seed}.")
                objective.backward()
                optimizer.step()
                total_train_bce += float(bce.detach().cpu()) * len(y_batch)
                observations += len(y_batch)

            train_bce = total_train_bce / observations
            self.model.eval()
            with torch.no_grad():
                validation_bce = float(loss_function(self.model(X_val_tensor), y_val_tensor).cpu())
            self.train_history.append({
                "epoch": epoch,
                "train_bce": train_bce,
                "validation_bce": validation_bce,
            })

            if validation_bce < best_loss - 1e-5:
                best_loss = validation_bce
                self.best_epoch = epoch
                stale_epochs = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
            else:
                stale_epochs += 1
                if verbose:
                    print(f"Época {epoch}: train BCE={train_bce:.5f}, val BCE={validation_bce:.5f}")
                if stale_epochs >= int(patience):
                    break

        if best_state is None:
            raise FloatingPointError(f"No se obtuvo un estado válido para la semilla {self.seed}.")
        self.model.load_state_dict(best_state)
        self.model.to(self.device).eval()
        self.validation_bce = float(best_loss)
        self.epochs_run = len(self.train_history)
        self.fit_time_sec = time.perf_counter() - started
        return self

    def predict_proba(self, X, batch_size: int = 8192) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("El modelo debe entrenarse antes de predecir.")
        X = matriz_densa_float32(X)
        self.model.eval()
        positive = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                batch = torch.from_numpy(X[start : start + batch_size]).to(self.device)
                positive.append(torch.sigmoid(self.model(batch)).cpu().numpy())
        probability = np.concatenate(positive)
        return np.column_stack([1.0 - probability, probability])

    def evaluate(self, X, y) -> dict[str, float]:
        """Evalúa la clase positiva con AUC, Gini, KS, Brier y log-loss."""
        from credit_risk.pd.evaluation import calcular_metricas_pd

        return calcular_metricas_pd(y, self.predict_proba(X)[:, 1])

    def save(self, filepath: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("No hay un modelo entrenado para guardar.")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
            "model_config": self.config,
            "input_dim": int(self.input_dim),
            "seed": self.seed,
            "best_epoch": self.best_epoch,
            "positive_class": 1,
        }, filepath)

    @classmethod
    def load(cls, filepath: str | Path, device=None):
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=True)
        if checkpoint.get("positive_class") != 1:
            raise ValueError("El checkpoint PD debe declarar positive_class=1.")
        instance = cls(checkpoint["model_config"], checkpoint["seed"], device=device)
        instance.input_dim = int(checkpoint["input_dim"])
        instance.best_epoch = int(checkpoint["best_epoch"])
        instance.model = PDNet(instance.input_dim, instance.config).to(instance.device)
        instance.model.load_state_dict(checkpoint["state_dict"])
        instance.model.eval()
        return instance
