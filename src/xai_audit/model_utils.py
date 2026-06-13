from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .interfaces import TabularBlackBoxModel
from .types import PredictionResult, TabularRow


def _normalize_probabilities(model: TabularBlackBoxModel, raw_probs: Any) -> dict[str, float] | None:
    if raw_probs is None:
        return None

    if isinstance(raw_probs, Mapping):
        return {str(key): float(value) for key, value in raw_probs.items()}

    if isinstance(raw_probs, Sequence) and not isinstance(raw_probs, (str, bytes)):
        classes = getattr(model, "classes_", None)
        if classes is None:
            classes = list(range(len(raw_probs)))
        return {str(label): float(prob) for label, prob in zip(classes, raw_probs, strict=False)}

    return None


def predict_single(
    model: TabularBlackBoxModel,
    instance: TabularRow,
    *,
    focus_label: str | None = None,
) -> PredictionResult:
    predicted = model.predict([instance])[0]

    raw_probabilities = None
    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            raw_probabilities = model.predict_proba([instance])[0]
            probabilities = _normalize_probabilities(model, raw_probabilities)
        except NotImplementedError:
            probabilities = None

    label = str(predicted)
    if probabilities:
        if focus_label is not None and focus_label in probabilities:
            score = probabilities[focus_label]
        elif label in probabilities:
            score = probabilities[label]
        else:
            best_label, score = max(probabilities.items(), key=lambda item: item[1])
            label = best_label
    elif isinstance(predicted, bool):
        score = float(int(predicted))
    elif isinstance(predicted, (int, float)):
        score = float(predicted)
    else:
        score = 1.0

    return PredictionResult(
        label=label,
        score=float(score),
        class_probabilities=probabilities,
        raw_output=raw_probabilities if raw_probabilities is not None else predicted,
    )


def is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
