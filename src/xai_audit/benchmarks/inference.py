from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from .training import TabularMLP, _to_dense_float32


@dataclass
class TrainedModelBundle:
    dataset_name: str
    artifact_dir: str
    model: TabularMLP
    preprocessor: Any
    metadata: dict[str, Any]

    @property
    def feature_names(self) -> list[str]:
        return list(self.metadata["feature_names"])

    @property
    def class_names(self) -> list[str]:
        return list(self.metadata.get("class_names", ["negative", "positive"]))

    def predict_proba(self, raw_features: pd.DataFrame | np.ndarray) -> np.ndarray:
        features_df = _coerce_feature_frame(raw_features, self.feature_names)
        transformed = _to_dense_float32(self.preprocessor.transform(features_df))
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(transformed))
            positive_probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
        negative_probabilities = 1.0 - positive_probabilities
        return np.column_stack([negative_probabilities, positive_probabilities])

    def predict(self, raw_features: pd.DataFrame | np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(raw_features)[:, 1]
        return (probabilities >= 0.5).astype(np.int64)


def load_trained_model_bundle(
    *,
    dataset_name: str,
    artifact_root: str = "artifacts/benchmark_models",
) -> TrainedModelBundle:
    artifact_dir = Path(artifact_root) / dataset_name
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(artifact_dir / "model.pt", map_location="cpu")
    model = TabularMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_layer_sizes=tuple(checkpoint["hidden_layer_sizes"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    preprocessor = joblib.load(artifact_dir / "preprocessor.joblib")
    return TrainedModelBundle(
        dataset_name=dataset_name,
        artifact_dir=str(artifact_dir.resolve()),
        model=model,
        preprocessor=preprocessor,
        metadata=metadata,
    )


def _coerce_feature_frame(raw_features: pd.DataFrame | np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    if isinstance(raw_features, pd.DataFrame):
        return raw_features[feature_names].copy()
    return pd.DataFrame(raw_features, columns=feature_names)
