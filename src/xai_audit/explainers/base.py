from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd

from xai_audit.benchmarks.inference import TrainedModelBundle
from xai_audit.interfaces import FixedTabularExplainer, TabularBlackBoxModel
from xai_audit.types import LocalExplanation, TabularRow


@dataclass
class ExplainerContext:
    bundle: TrainedModelBundle
    background_data: pd.DataFrame
    feature_names: list[str]
    class_names: list[str]
    top_k_features: int = 10
    random_state: int = 42

    @property
    def categorical_feature_names(self) -> list[str]:
        return list(self.bundle.metadata.get("categorical_features", []))

    @property
    def categorical_feature_indices(self) -> list[int]:
        feature_index = {name: idx for idx, name in enumerate(self.feature_names)}
        return [feature_index[name] for name in self.categorical_feature_names if name in feature_index]

    @property
    def categorical_names(self) -> dict[int, list[str]]:
        feature_index = {name: idx for idx, name in enumerate(self.feature_names)}
        output: dict[int, list[str]] = {}
        for feature_name in self.categorical_feature_names:
            if feature_name not in self.background_data.columns:
                continue
            output[feature_index[feature_name]] = [
                str(value)
                for value in self.background_data[feature_name].dropna().astype(str).unique().tolist()
            ]
        return output

    def predict_positive_probability(self, values: pd.DataFrame | Any) -> Any:
        return self.bundle.predict_proba(_coerce_frame(values, self.feature_names))[:, 1]

    def predict_class_probabilities(self, values: pd.DataFrame | Any) -> Any:
        return self.bundle.predict_proba(_coerce_frame(values, self.feature_names))

    def lime_training_data(self) -> Any:
        return self.encode_for_lime(self.background_data).to_numpy()

    def encode_for_lime(self, frame: pd.DataFrame) -> pd.DataFrame:
        encoded = frame[self.feature_names].copy()
        feature_index = {name: idx for idx, name in enumerate(self.feature_names)}
        for feature_name in self.categorical_feature_names:
            if feature_name not in encoded.columns:
                continue
            mapping = {
                category: idx
                for idx, category in enumerate(self.categorical_names[feature_index[feature_name]])
            }
            encoded[feature_name] = (
                encoded[feature_name]
                .astype(str)
                .map(mapping)
                .fillna(0)
                .astype(int)
            )
        return encoded

    def decode_lime_array(self, values: Any) -> pd.DataFrame:
        decoded = pd.DataFrame(values, columns=self.feature_names)
        feature_index = {name: idx for idx, name in enumerate(self.feature_names)}
        for feature_name in self.categorical_feature_names:
            if feature_name not in decoded.columns:
                continue
            categories = self.categorical_names[feature_index[feature_name]]
            decoded[feature_name] = (
                decoded[feature_name]
                .round()
                .astype(int)
                .clip(lower=0, upper=max(len(categories) - 1, 0))
                .map(lambda idx: categories[idx] if categories else "")
            )
        return decoded


@dataclass
class ExplainedFeature:
    feature_name: str
    attribution: float
    rank: int


class BaseAuditExplainer(FixedTabularExplainer, ABC):
    name: str

    def __init__(self, *, context: ExplainerContext) -> None:
        self.context = context

    @abstractmethod
    def explain_instance(
        self,
        model: TabularBlackBoxModel,
        instance: TabularRow,
        *,
        reference_row: TabularRow | None = None,
        focus_label: str | None = None,
    ) -> LocalExplanation:
        raise NotImplementedError


_EXPLAINER_REGISTRY: dict[str, type[BaseAuditExplainer]] = {}


def register_explainer(name: str):
    def decorator(cls: type[BaseAuditExplainer]) -> type[BaseAuditExplainer]:
        _EXPLAINER_REGISTRY[name] = cls
        cls.name = name
        return cls

    return decorator


def available_explainers() -> list[str]:
    return sorted(_EXPLAINER_REGISTRY)


def build_explainer(name: str, *, context: ExplainerContext, **kwargs: Any) -> BaseAuditExplainer:
    if name not in _EXPLAINER_REGISTRY:
        raise KeyError(f"Unknown explainer {name!r}. Available: {', '.join(available_explainers())}")
    return _EXPLAINER_REGISTRY[name](context=context, **kwargs)


def top_attributions(feature_names: list[str], values: list[float] | Any, top_k: int) -> dict[str, float]:
    import numpy as np

    array = np.asarray(values, dtype=float)
    indices = np.argsort(np.abs(array))[::-1][:top_k]
    return {feature_names[int(index)]: float(array[int(index)]) for index in indices}


def _coerce_frame(values: pd.DataFrame | Any, feature_names: list[str]) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values[feature_names].copy()
    return pd.DataFrame(values, columns=feature_names)
