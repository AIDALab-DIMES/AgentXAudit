from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from random import Random

from .interfaces import FixedTabularExplainer, TabularBlackBoxModel
from .model_utils import is_numeric, predict_single
from .types import LocalExplanation, TabularRow, TechnicalAssessment


@dataclass
class TechnicalAssessmentConfig:
    top_k_features: int = 3
    directional_step_fraction: float = 0.05
    stability_samples: int = 8
    stability_noise_fraction: float = 0.03
    random_seed: int = 7


class TechnicalAssessor:
    def __init__(self, config: TechnicalAssessmentConfig | None = None) -> None:
        self.config = config or TechnicalAssessmentConfig()

    def assess(
        self,
        *,
        model: TabularBlackBoxModel,
        explainer: FixedTabularExplainer,
        instance: TabularRow,
        explanation: LocalExplanation,
        reference_row: TabularRow | None = None,
        focus_label: str | None = None,
    ) -> TechnicalAssessment:
        notes: list[str] = []
        diagnostics: dict[str, float | int | str] = {}

        reconstruction_fidelity = self._reconstruction_fidelity(explanation, diagnostics)
        if reconstruction_fidelity is None:
            notes.append("Reconstruction fidelity was skipped because no additive baseline was available.")

        directional_faithfulness = self._directional_faithfulness(
            model=model,
            instance=instance,
            explanation=explanation,
            focus_label=focus_label,
            diagnostics=diagnostics,
        )
        if directional_faithfulness is None:
            notes.append("Directional faithfulness was skipped because no numeric attributed features were available.")

        stability = self._stability(
            model=model,
            explainer=explainer,
            instance=instance,
            explanation=explanation,
            reference_row=reference_row,
            focus_label=focus_label,
            diagnostics=diagnostics,
        )
        if stability is None:
            notes.append("Stability was skipped because local perturbations could not be generated.")

        return TechnicalAssessment(
            reconstruction_fidelity=reconstruction_fidelity,
            directional_faithfulness=directional_faithfulness,
            stability=stability,
            notes=notes,
            diagnostics=diagnostics,
        )

    def _reconstruction_fidelity(
        self,
        explanation: LocalExplanation,
        diagnostics: dict[str, float | int | str],
    ) -> float | None:
        if explanation.baseline_value is None:
            return None

        reconstructed = explanation.baseline_value + sum(explanation.feature_attributions.values())
        prediction = explanation.prediction.score
        error = abs(prediction - reconstructed)
        scale = max(1.0, abs(prediction), abs(reconstructed))
        score = max(0.0, 1.0 - (error / scale))

        diagnostics["reconstructed_score"] = round(reconstructed, 6)
        diagnostics["reconstruction_error"] = round(error, 6)
        return round(score, 4)

    def _directional_faithfulness(
        self,
        *,
        model: TabularBlackBoxModel,
        instance: TabularRow,
        explanation: LocalExplanation,
        focus_label: str | None,
        diagnostics: dict[str, float | int | str],
    ) -> float | None:
        ranked_features = [
            (feature, value)
            for feature, value in sorted(
                explanation.feature_attributions.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
            if is_numeric(instance.get(feature))
        ]
        ranked_features = ranked_features[: self.config.top_k_features]
        if not ranked_features:
            return None

        base_score = explanation.prediction.score
        local_effects = explanation.metadata.get("local_effects", {})
        agreements = 0
        measured = 0

        for feature, attribution in ranked_features:
            expected_effect = float(local_effects.get(feature, attribution))
            if expected_effect == 0:
                continue
            original_value = float(instance[feature])
            step = max(abs(original_value), 1.0) * self.config.directional_step_fraction
            perturbed = dict(instance)
            perturbed[feature] = original_value + step
            new_score = predict_single(model, perturbed, focus_label=focus_label).score
            delta = new_score - base_score
            measured += 1

            if delta == 0:
                agreements += 0.5
            elif delta * expected_effect > 0:
                agreements += 1

        if measured == 0:
            return None

        score = agreements / measured
        diagnostics["directional_features_tested"] = measured
        return round(score, 4)

    def _stability(
        self,
        *,
        model: TabularBlackBoxModel,
        explainer: FixedTabularExplainer,
        instance: TabularRow,
        explanation: LocalExplanation,
        reference_row: TabularRow | None,
        focus_label: str | None,
        diagnostics: dict[str, float | int | str],
    ) -> float | None:
        numeric_features = [feature for feature, value in instance.items() if is_numeric(value)]
        if not numeric_features:
            return None

        rng = Random(self.config.random_seed)
        original_vector = explanation.feature_attributions
        similarities: list[float] = []

        for _ in range(self.config.stability_samples):
            perturbed = dict(instance)
            for feature in numeric_features:
                base = float(instance[feature])
                noise_scale = max(abs(base), 1.0) * self.config.stability_noise_fraction
                perturbed[feature] = base + rng.uniform(-noise_scale, noise_scale)

            perturbed_explanation = explainer.explain_instance(
                model,
                perturbed,
                reference_row=reference_row,
                focus_label=focus_label,
            )
            cosine = _cosine_similarity(original_vector, perturbed_explanation.feature_attributions)
            overlap = _top_k_overlap(
                original_vector,
                perturbed_explanation.feature_attributions,
                self.config.top_k_features,
            )
            similarities.append((cosine + overlap) / 2.0)

        if not similarities:
            return None

        diagnostics["stability_samples"] = len(similarities)
        return round(sum(similarities) / len(similarities), 4)


def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    features = sorted(set(left) | set(right))
    if not features:
        return 0.0

    numerator = sum(left.get(feature, 0.0) * right.get(feature, 0.0) for feature in features)
    left_norm = sqrt(sum(left.get(feature, 0.0) ** 2 for feature in features))
    right_norm = sqrt(sum(right.get(feature, 0.0) ** 2 for feature in features))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, min(1.0, numerator / (left_norm * right_norm)))


def _top_k_overlap(left: dict[str, float], right: dict[str, float], top_k: int) -> float:
    left_top = {
        feature
        for feature, _ in sorted(left.items(), key=lambda item: abs(item[1]), reverse=True)[:top_k]
    }
    right_top = {
        feature
        for feature, _ in sorted(right.items(), key=lambda item: abs(item[1]), reverse=True)[:top_k]
    }
    if not left_top and not right_top:
        return 1.0
    if not left_top or not right_top:
        return 0.0
    return len(left_top & right_top) / len(left_top | right_top)
