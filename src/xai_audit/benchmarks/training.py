from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .datasets import DATASET_SPECS, LoadedTabularDataset, available_dataset_names, load_dataset


@dataclass(frozen=True)
class NeuralModelConfig:
    hidden_layer_sizes: tuple[int, ...]
    alpha: float = 1e-4
    learning_rate_init: float = 1e-3
    batch_size: int = 64
    max_iter: int = 400
    early_stopping: bool = True
    n_iter_no_change: int = 20


MODEL_CONFIGS: dict[str, NeuralModelConfig] = {
    "german_credit": NeuralModelConfig(hidden_layer_sizes=(64, 32), batch_size=32, max_iter=500),
    "credit_card_default": NeuralModelConfig(hidden_layer_sizes=(128, 64, 32), batch_size=256, max_iter=300),
    "heart_disease": NeuralModelConfig(hidden_layer_sizes=(32, 16), batch_size=32, max_iter=500),
    "breast_cancer": NeuralModelConfig(hidden_layer_sizes=(64, 32), batch_size=32, max_iter=500),
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

    config = MODEL_CONFIGS[dataset_name]
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.features,
        dataset.target,
        test_size=0.2,
        random_state=random_state,
        stratify=dataset.target,
    )

    pipeline = _build_pipeline(dataset, config, random_state=random_state)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
        "f1": round(float(f1_score(y_test, y_pred)), 6),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 6),
        "average_precision": round(float(average_precision_score(y_test, y_proba)), 6),
    }

    artifact_root = Path(output_dir) / dataset_name
    artifact_root.mkdir(parents=True, exist_ok=True)

    model_path = artifact_root / "model.joblib"
    metrics_path = artifact_root / "metrics.json"
    metadata_path = artifact_root / "metadata.json"
    dataset_csv_path = artifact_root / "dataset.csv"
    split_csv_path = artifact_root / "test_split.csv"

    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    metadata = {
        **dataset.metadata,
        "dataset_name": dataset_name,
        "model_family": "sklearn_mlp_classifier",
        "model_config": asdict(config),
        "artifact_files": {
            "model": str(model_path.resolve()),
            "metrics": str(metrics_path.resolve()),
            "metadata": str(metadata_path.resolve()),
            "dataset_csv": str(dataset_csv_path.resolve()),
            "test_split_csv": str(split_csv_path.resolve()),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    dataset_export = dataset.features.copy()
    dataset_export["target"] = dataset.target.values
    dataset_export.to_csv(dataset_csv_path, index=False)

    test_export = X_test.copy()
    test_export["target"] = y_test.values
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


def _build_pipeline(
    dataset: LoadedTabularDataset,
    config: NeuralModelConfig,
    *,
    random_state: int,
) -> Pipeline:
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

    preprocessor = ColumnTransformer(transformers=transformers)
    model = MLPClassifier(
        hidden_layer_sizes=config.hidden_layer_sizes,
        alpha=config.alpha,
        learning_rate_init=config.learning_rate_init,
        batch_size=config.batch_size,
        max_iter=config.max_iter,
        early_stopping=config.early_stopping,
        n_iter_no_change=config.n_iter_no_change,
        random_state=random_state,
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
