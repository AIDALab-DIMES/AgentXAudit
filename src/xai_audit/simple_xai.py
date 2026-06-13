from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Sequence

from .interfaces import TabularBlackBoxModel
from .model_utils import is_numeric, predict_single
from .types import LocalExplanation, TabularRow


@dataclass
class FiniteDifferenceTabularExplainer:
    name: str = "finite_difference_local_additive"
    finite_difference_fraction: float = 0.02
    rescale_to_prediction_gap: bool = True

    def explain_instance(
        self,
        model: TabularBlackBoxModel,
        instance: TabularRow,
        *,
        reference_row: TabularRow | None = None,
        focus_label: str | None = None,
    ) -> LocalExplanation:
        prediction = predict_single(model, instance, focus_label=focus_label)
        reference = reference_row or _zero_reference(instance)
        baseline_prediction = predict_single(model, reference, focus_label=focus_label)

        attributions: dict[str, float] = {}
        local_effects: dict[str, float] = {}
        skipped_features: list[str] = []
        for feature, value in instance.items():
            reference_value = reference.get(feature)
            if not (is_numeric(value) and is_numeric(reference_value)):
                skipped_features.append(feature)
                continue

            local_effect = self._local_effect(
                model=model,
                instance=instance,
                feature=feature,
                focus_label=focus_label,
            )
            local_effects[feature] = local_effect
            delta_from_reference = float(value) - float(reference_value)
            attributions[feature] = local_effect * delta_from_reference

        if self.rescale_to_prediction_gap and attributions:
            raw_sum = sum(attributions.values())
            target_gap = prediction.score - baseline_prediction.score
            if raw_sum != 0:
                scale = target_gap / raw_sum
                attributions = {feature: value * scale for feature, value in attributions.items()}

        explanation_text = _render_explanation_text(attributions, prediction.label)
        metadata = {
            "reference_row": reference,
            "skipped_features": skipped_features,
            "rescaled_to_prediction_gap": self.rescale_to_prediction_gap,
            "local_effects": {feature: round(value, 6) for feature, value in local_effects.items()},
        }

        return LocalExplanation(
            method_name=self.name,
            prediction=prediction,
            feature_attributions={feature: round(value, 6) for feature, value in attributions.items()},
            baseline_value=round(baseline_prediction.score, 6),
            explanation_text=explanation_text,
            metadata=metadata,
        )

    def _local_effect(
        self,
        *,
        model: TabularBlackBoxModel,
        instance: TabularRow,
        feature: str,
        focus_label: str | None,
    ) -> float:
        value = float(instance[feature])
        delta = max(abs(value), 1.0) * self.finite_difference_fraction

        plus = dict(instance)
        minus = dict(instance)
        plus[feature] = value + delta
        minus[feature] = value - delta

        plus_score = predict_single(model, plus, focus_label=focus_label).score
        minus_score = predict_single(model, minus, focus_label=focus_label).score
        return (plus_score - minus_score) / (2.0 * delta)


class ToyCreditRiskModel:
    name = "toy_credit_risk_model"
    feature_names: Sequence[str] = (
        "income_k",
        "debt_ratio",
        "savings_k",
        "missed_payments",
        "age",
    )
    classes_ = ("reject", "approve")

    def predict(self, rows: Sequence[TabularRow]) -> list[str]:
        return [
            "approve" if self._approve_probability(row) >= 0.5 else "reject"
            for row in rows
        ]

    def predict_proba(self, rows: Sequence[TabularRow]) -> list[dict[str, float]]:
        outputs: list[dict[str, float]] = []
        for row in rows:
            approve = self._approve_probability(row)
            outputs.append({"reject": 1.0 - approve, "approve": approve})
        return outputs

    def _approve_probability(self, row: TabularRow) -> float:
        income_k = float(row["income_k"])
        debt_ratio = float(row["debt_ratio"])
        savings_k = float(row["savings_k"])
        missed_payments = float(row["missed_payments"])
        age = float(row["age"])

        logit = (
            -1.2
            + (0.04 * income_k)
            - (3.0 * debt_ratio)
            + (0.06 * savings_k)
            - (0.7 * missed_payments)
            + (0.015 * age)
        )
        return 1.0 / (1.0 + exp(-logit))


def _zero_reference(instance: TabularRow) -> TabularRow:
    reference: TabularRow = {}
    for feature, value in instance.items():
        reference[feature] = 0.0 if is_numeric(value) else value
    return reference


def _render_explanation_text(attributions: dict[str, float], label: str) -> str:
    if not attributions:
        return "The explainer could not derive numeric feature attributions for this instance."

    ranked = sorted(attributions.items(), key=lambda item: abs(item[1]), reverse=True)
    top = ranked[:3]
    clauses = []
    for feature, contribution in top:
        direction = "increases" if contribution > 0 else "decreases"
        clauses.append(f"{feature} {direction} the score by {abs(contribution):.3f}")
    joined = "; ".join(clauses)
    return f"Predicted label: {label}. Main local factors: {joined}."
