"""Red neuronal (Perceptrón Multicapa - MLP) para Pérdida Dado el Incumplimiento (LGD).

Justificación de diseño:
1. Capa de salida con activación Sigmoide (logistic):
   La LGD representa la fracción económica de pérdida sobre la exposición impagada,
   por lo que su dominio teórico y empírico está acotado en el intervalo [0, 1].
   Una capa lineal sin acotamiento puede generar predicciones negativas (recuperación mayor
   al 100%) o mayores a 1 (pérdidas que exceden el capital), lo cual carece de sentido financiero.
   La activación Sigmoide sigma(z) = 1 / (1 + exp(-z)) garantiza por construcción matemática
   que la predicción y_hat in (0, 1) sin requerir truncamientos artificiales a posteriori.

2. Función de pérdida:
   - Error Cuadrático Medio (MSE): Se deriva mediante Estimación de Máxima Verosimilitud (MLE)
     bajo el supuesto de que los errores epsilon_i = y_i - y_hat_i siguen una distribución
     normal condicional N(0, sigma^2). Maximizar el logaritmo de verosimilitud equivale a
     minimizar la suma de cuadrados residuales.
   - Entropía Cruzada Binaria (BCE / Fractional Quasi-Likelihood): Como propone Papke & Wooldridge (1996),
     cuando la variable dependiente es una proporción continua y in [0, 1], la función de quasi-verosimilitud
     de Bernoulli provee estimadores consistentes para la media condicional E[y|x].
   - Huber Loss (Smooth L1): Ofrece robustez cuadrática para errores pequeños y lineal para errores grandes,
     protegiendo el gradiente contra valores extremos de recuperación.
"""
import time
from pathlib import Path
import numpy as np
from scipy import sparse
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error


class MLPRegressor(nn.Module):
    """Arquitectura de red neuronal para regresión de variables acotadas [0, 1]."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = None,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        activation: str = "relu",
        output_activation: str = "sigmoid",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        act_cls = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU,
        }.get(activation.lower(), nn.ReLU)

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Capa de salida lineal
        layers.append(nn.Linear(prev_dim, 1))

        # Activación de salida
        if output_activation.lower() == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation.lower() != "linear":
            raise ValueError(f"Activación de salida desconocida: {output_activation}")

        self.network = nn.Sequential(*layers)
        self.output_activation = output_activation.lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        if self.output_activation == "linear":
            out = torch.clamp(out, 0.0, 1.0)
        return out


class LGDNeuralNetwork:
    """Envoltorio de alto nivel para entrenamiento, validación e inferencia de LGD con PyTorch."""

    def __init__(
        self,
        hidden_dims: list[int] = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_function: str = "mse",
        batch_size: int = 128,
        epochs: int = 100,
        patience: int = 10,
        use_batch_norm: bool = True,
        activation: str = "relu",
        output_activation: str = "sigmoid",
        device: str = None,
        seed: int = 42,
    ):
        self.hidden_dims = hidden_dims or [64, 32]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_function = loss_function.lower()
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.output_activation = output_activation
        self.seed = seed

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: MLPRegressor = None
        self.input_dim: int = None
        self.train_history = []

    @property
    def n_parameters(self) -> int:
        """Número de parámetros entrenables de la red."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _get_loss_fn(self):
        if self.loss_function == "mse":
            return nn.MSELoss()
        elif self.loss_function in {"bce", "binary_cross_entropy"}:
            return nn.BCELoss()
        elif self.loss_function in {"huber", "smooth_l1"}:
            return nn.HuberLoss()
        elif self.loss_function in {"mae", "l1"}:
            return nn.L1Loss()
        else:
            raise ValueError(f"Función de pérdida desconocida: {self.loss_function}")

    def _to_tensor(self, X, y=None):
        if sparse.issparse(X):
            X_arr = X.toarray()
        elif isinstance(X, np.ndarray):
            X_arr = X
        else:
            X_arr = np.asarray(X)

        t_X = torch.tensor(X_arr, dtype=torch.float32, device=self.device)
        if y is not None:
            t_y = torch.tensor(y, dtype=torch.float32, device=self.device).view(-1, 1)
            return t_X, t_y
        return t_X

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose: bool = False):
        """Entrena la red con parada temprana basada en el error de validación."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.input_dim = X_train.shape[1]
        self.model = MLPRegressor(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            use_batch_norm=self.use_batch_norm,
            activation=self.activation,
            output_activation=self.output_activation,
        ).to(self.device)

        loss_fn = self._get_loss_fn()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        t_X_train, t_y_train = self._to_tensor(X_train, y_train)
        dataset = torch.utils.data.TensorDataset(t_X_train, t_y_train)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        if X_val is not None and y_val is not None:
            t_X_val, t_y_val = self._to_tensor(X_val, y_val)
        else:
            t_X_val, t_y_val = None, None

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        self.train_history = []

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss_sum = 0.0
            n_batches = 0

            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = loss_fn(preds, batch_y)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()
                n_batches += 1

            train_loss = train_loss_sum / max(n_batches, 1)

            # Validación
            if t_X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(t_X_val)
                    val_loss = loss_fn(val_preds, t_y_val).item()

                self.train_history.append({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                })

                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if verbose and (epoch % 10 == 0 or epoch == 1):
                    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

                if patience_counter >= self.patience:
                    if verbose:
                        print(f"Early stopping activado en epoch {epoch}. Mejor Val Loss: {best_val_loss:.5f}")
                    break
            else:
                self.train_history.append({"epoch": epoch, "train_loss": train_loss})

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.fit_time_sec = time.time() - start_time
        return self

    def predict(self, X) -> np.ndarray:
        """Predice LGD acotada en [0, 1]."""
        if self.model is None:
            raise RuntimeError("El modelo debe ser entrenado antes de predecir.")
        self.model.eval()
        with torch.no_grad():
            t_X = self._to_tensor(X)
            preds = self.model(t_X).view(-1).cpu().numpy()
        return np.clip(preds, 0.0, 1.0)

    def evaluate(self, X, y) -> tuple[float, float]:
        """Calcula MAE y RMSE sobre un conjunto de evaluación."""
        preds = self.predict(X)
        mae = float(mean_absolute_error(y, preds))
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        return mae, rmse

    def save(self, filepath: str | Path):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "loss_function": self.loss_function,
            "use_batch_norm": self.use_batch_norm,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "seed": self.seed,
            "train_history": self.train_history,
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load(cls, filepath: str | Path, device: str = None):
        checkpoint = torch.load(filepath, map_location="cpu")
        instance = cls(
            hidden_dims=checkpoint["hidden_dims"],
            dropout=checkpoint["dropout"],
            learning_rate=checkpoint["learning_rate"],
            weight_decay=checkpoint["weight_decay"],
            loss_function=checkpoint["loss_function"],
            use_batch_norm=checkpoint["use_batch_norm"],
            activation=checkpoint["activation"],
            output_activation=checkpoint["output_activation"],
            device=device,
            seed=checkpoint.get("seed", 42),
        )
        instance.input_dim = checkpoint["input_dim"]
        instance.model = MLPRegressor(
            input_dim=instance.input_dim,
            hidden_dims=instance.hidden_dims,
            dropout=instance.dropout,
            use_batch_norm=instance.use_batch_norm,
            activation=instance.activation,
            output_activation=instance.output_activation,
        ).to(instance.device)
        instance.model.load_state_dict(checkpoint["state_dict"])
        instance.train_history = checkpoint.get("train_history", [])
        return instance
