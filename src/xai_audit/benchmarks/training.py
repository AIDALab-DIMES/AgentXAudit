from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .datasets import LoadedTabularDataset, available_dataset_names, load_dataset


@dataclass(frozen=True)
class NeuralModelConfig:
    hidden_layer_sizes: tuple[int, ...]
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 120
    patience: int = 15
    validation_fraction: float = 0.15


MODEL_CONFIGS: dict[str, NeuralModelConfig] = {
    "german_credit": NeuralModelConfig(hidden_layer_sizes=(64, 32), batch_size=32, epochs=160, patience=20),
    "credit_card_default": NeuralModelConfig(hidden_layer_sizes=(128, 64, 32), batch_size=256, epochs=80, patience=12),
    "heart_disease": NeuralModelConfig(hidden_layer_sizes=(32, 16), batch_size=32, epochs=180, patience=25),
    "breast_cancer": NeuralModelConfig(hidden_layer_sizes=(64, 32), batch_size=32, epochs=160, patience=20),
}


@dataclass
class TrainingResult:
    dataset_name: str
    domain: str
    artifact_dir: str
    model_path: str
    metrics_path: str
    dataset_csv_path: str
    metrics: dict[str, float]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layer_sizes: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


def train_many_benchmark_datasets(
    *,
    dataset_names: list[str] | None = None,
    output_dir: str = "artifacts/benchmark_models",
    random_state: int = 42,
) -> list[TrainingResult]:
    names = dataset_names or available_dataset_names()
    return [
        train_benchmark_dataset(
            dataset_name=name,
            output_dir=output_dir,
            random_state=random_state,
        )
        for name in names
    ]


def train_benchmark_dataset(
    *,
    dataset_name: str,
    output_dir: str = "artifacts/benchmark_models",
    random_state: int = 42,
) -> TrainingResult:
    dataset = load_dataset(dataset_name)
    if dataset_name not in MODEL_CONFIGS:
        raise KeyError(f"No neural model config registered for {dataset_name!r}.")

    _set_random_seed(random_state)
    config = MODEL_CONFIGS[dataset_name]
    X_train_df, X_test_df, y_train_series, y_test_series = train_test_split(
        dataset.features,
        dataset.target,
        test_size=0.2,
        random_state=random_state,
        stratify=dataset.target,
    )
    X_fit_df, X_val_df, y_fit_series, y_val_series = train_test_split(
        X_train_df,
        y_train_series,
        test_size=config.validation_fraction,
        random_state=random_state,
        stratify=y_train_series,
    )

    preprocessor = _build_preprocessor(dataset)
    X_fit = _to_dense_float32(preprocessor.fit_transform(X_fit_df))
    X_val = _to_dense_float32(preprocessor.transform(X_val_df))
    X_test = _to_dense_float32(preprocessor.transform(X_test_df))

    y_fit = y_fit_series.to_numpy(dtype=np.float32)
    y_val = y_val_series.to_numpy(dtype=np.float32)
    y_test = y_test_series.to_numpy(dtype=np.float32)

    model = TabularMLP(
        input_dim=X_fit.shape[1],
        hidden_layer_sizes=config.hidden_layer_sizes,
        dropout=config.dropout,
    )
    history = _train_torch_model(
        model=model,
        X_fit=X_fit,
        y_fit=y_fit,
        X_val=X_val,
        y_val=y_val,
        config=config,
    )

    y_proba = _predict_probabilities(model, X_test)
    y_pred = (y_proba >= 0.5).astype(np.int64)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
        "f1": round(float(f1_score(y_test, y_pred)), 6),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 6),
        "average_precision": round(float(average_precision_score(y_test, y_proba)), 6),
    }

    artifact_root = Path(output_dir) / dataset_name
    artifact_root.mkdir(parents=True, exist_ok=True)

    model_path = artifact_root / "model.pt"
    preprocessor_path = artifact_root / "preprocessor.joblib"
    metrics_path = artifact_root / "metrics.json"
    metadata_path = artifact_root / "metadata.json"
    dataset_csv_path = artifact_root / "dataset.csv"
    split_csv_path = artifact_root / "test_split.csv"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": X_fit.shape[1],
            "hidden_layer_sizes": config.hidden_layer_sizes,
            "dropout": config.dropout,
        },
        model_path,
    )
    joblib.dump(preprocessor, preprocessor_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    metadata = {
        **dataset.metadata,
        "dataset_name": dataset_name,
        "model_family": "pytorch_tabular_mlp",
        "model_config": asdict(config),
        "random_state": random_state,
        "feature_names": list(dataset.features.columns),
        "categorical_features": list(dataset.categorical_features),
        "numeric_features": list(dataset.numeric_features),
        "class_names": ["negative", dataset.spec.positive_class],
        "training_history": history,
        "artifact_files": {
            "model": str(model_path.resolve()),
            "preprocessor": str(preprocessor_path.resolve()),
            "metrics": str(metrics_path.resolve()),
            "metadata": str(metadata_path.resolve()),
            "dataset_csv": str(dataset_csv_path.resolve()),
            "test_split_csv": str(split_csv_path.resolve()),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    dataset_export = dataset.features.copy()
    dataset_export.insert(0, "source_row_id", dataset.features.index.astype(str))
    dataset_export["target"] = dataset.target.values
    dataset_export.to_csv(dataset_csv_path, index=False)

    test_export = X_test_df.copy()
    test_export.insert(0, "source_row_id", X_test_df.index.astype(str))
    test_export["target"] = y_test_series.values
    test_export["predicted_probability"] = y_proba
    test_export["predicted_label"] = y_pred
    test_export.to_csv(split_csv_path, index=False)

    return TrainingResult(
        dataset_name=dataset_name,
        domain=dataset.spec.domain,
        artifact_dir=str(artifact_root.resolve()),
        model_path=str(model_path.resolve()),
        metrics_path=str(metrics_path.resolve()),
        dataset_csv_path=str(dataset_csv_path.resolve()),
        metrics=metrics,
        metadata=metadata,
    )


def _build_preprocessor(dataset: LoadedTabularDataset) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if dataset.numeric_features:
        transformers.append(("numeric", numeric_pipeline, dataset.numeric_features))
    if dataset.categorical_features:
        transformers.append(("categorical", categorical_pipeline, dataset.categorical_features))
    return ColumnTransformer(transformers=transformers)


def _train_torch_model(
    *,
    model: TabularMLP,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: NeuralModelConfig,
) -> dict[str, float | int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_fit),
            torch.from_numpy(y_fit),
        ),
        batch_size=config.batch_size,
        shuffle=True,
    )
    X_val_tensor = torch.from_numpy(X_val).to(device)
    y_val_tensor = torch.from_numpy(y_val).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            logits = model(batch_inputs)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_loss = float(criterion(val_logits, y_val_tensor).item())

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(torch.device("cpu"))

    return {
        "best_validation_loss": round(best_val_loss, 6),
        "best_epoch": int(best_epoch),
        "epochs_completed": int(epoch),
    }


def _predict_probabilities(model: TabularMLP, inputs: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(inputs))
        probabilities = torch.sigmoid(logits).cpu().numpy()
    return probabilities.astype(np.float64)


def _to_dense_float32(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
