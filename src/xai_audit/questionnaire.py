from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Literal, Sequence

from .interfaces import FixedTabularExplainer, TabularBlackBoxModel
from .model_utils import is_numeric, predict_single
from .types import PerturbationQuestion, SampleAuditRecord, TabularRow


@dataclass
class ValidationSamplingConfig:
    sample_count: int = 5
    min_sample_count: int = 5
    max_sample_count: int = 10
    perturbation_fraction: float = 0.15

    def validate(self) -> None:
        if not (self.min_sample_count <= self.sample_count <= self.max_sample_count):
            raise ValueError(
                f"sample_count must be between {self.min_sample_count} and {self.max_sample_count}; "
                f"received {self.sample_count}."
            )


def build_sample_audit_records(
    *,
    model: TabularBlackBoxModel,
    explainer: FixedTabularExplainer,
    validation_set: Sequence[TabularRow],
    config: ValidationSamplingConfig | None = None,
    reference_row: TabularRow | None = None,
    focus_label: str | None = None,
) -> list[SampleAuditRecord]:
    sampling_config = config or ValidationSamplingConfig()
    sampling_config.validate()

    if len(validation_set) < sampling_config.sample_count:
        raise ValueError(
            f"validation_set must contain at least {sampling_config.sample_count} rows; "
            f"received {len(validation_set)}."
        )

    selected_indices = select_validation_indices(
        model=model,
        validation_set=validation_set,
        sample_count=sampling_config.sample_count,
        focus_label=focus_label,
    )

    records: list[SampleAuditRecord] = []
    for record_id, index in enumerate(selected_indices, start=1):
        raw_row = dict(validation_set[index])
        instance, ground_truth_label = _split_instance_and_target(raw_row, model)
        prediction = predict_single(model, instance, focus_label=focus_label)
        explanation = explainer.explain_instance(
            model,
            instance,
            reference_row=reference_row,
            focus_label=focus_label,
        )
        question = build_perturbation_question(
            sample_id=f"sample_{record_id}",
            model=model,
            instance=instance,
            explanation=explanation,
            focus_label=focus_label,
            perturbation_fraction=sampling_config.perturbation_fraction,
        )
        records.append(
            SampleAuditRecord(
                sample_id=f"sample_{record_id}",
                instance=instance,
                prediction=prediction,
                explanation=explanation,
                ground_truth_label=ground_truth_label,
                prediction_is_correct=(
                    prediction.label == ground_truth_label if ground_truth_label is not None else None
                ),
                perturbation_questions=[question] if question is not None else [],
            )
        )
    return records


def select_validation_indices(
    *,
    model: TabularBlackBoxModel,
    validation_set: Sequence[TabularRow],
    sample_count: int,
    focus_label: str | None = None,
) -> list[int]:
    scored_rows = [
        (index, predict_single(model, _split_instance_and_target(row, model)[0], focus_label=focus_label).score)
        for index, row in enumerate(validation_set)
    ]
    ranked = sorted(scored_rows, key=lambda item: item[1])
    if sample_count >= len(ranked):
        return [index for index, _ in ranked]

    chosen_positions: list[int] = []
    max_position = len(ranked) - 1
    for slot in range(sample_count):
        raw_position = floor((slot * max_position) / (sample_count - 1)) if sample_count > 1 else 0
        if raw_position not in chosen_positions:
            chosen_positions.append(raw_position)

    probe = 0
    while len(chosen_positions) < sample_count:
        if probe not in chosen_positions:
            chosen_positions.append(probe)
        probe += 1

    return [ranked[position][0] for position in sorted(chosen_positions)]


def build_perturbation_question(
    *,
    sample_id: str,
    model: TabularBlackBoxModel,
    instance: TabularRow,
    explanation,
    focus_label: str | None,
    perturbation_fraction: float,
) -> PerturbationQuestion | None:
    ranked_features = [
        feature
        for feature, _ in sorted(
            explanation.feature_attributions.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        if is_numeric(instance.get(feature))
    ]
    if not ranked_features:
        return None

    feature = ranked_features[0]
    original_value = float(instance[feature])
    delta = max(abs(original_value), 1.0) * perturbation_fraction
    perturbed_value = original_value + delta

    perturbed_instance = dict(instance)
    perturbed_instance[feature] = round(perturbed_value, 6)
    perturbed_prediction = predict_single(model, perturbed_instance, focus_label=focus_label)
    base_score = explanation.prediction.score
    direction = _score_direction(perturbed_prediction.score - base_score)
    feature_change = f"{feature} increases from {original_value:.3f} to {perturbed_value:.3f}"
    prompt = (
        f"For {sample_id}, imagine the following perturbation: {feature_change}. "
        "Based on the model output and explanation shown for the original sample, "
        "what label would the black-box model predict now, and would the focal score increase, decrease, or stay about the same?"
    )

    return PerturbationQuestion(
        question_id=f"{sample_id}_q1",
        sample_id=sample_id,
        prompt=prompt,
        perturbed_instance=perturbed_instance,
        feature_name=feature,
        feature_change=feature_change,
        expected_label=perturbed_prediction.label,
        expected_score=round(perturbed_prediction.score, 6),
        expected_score_direction=direction,
    )


def user_visible_records(records: Sequence[SampleAuditRecord]) -> list[dict[str, object]]:
    return [
        {
            "sample_id": record.sample_id,
            "instance": record.instance,
            "model_output": record.prediction.to_dict(),
            "explanation_output": record.explanation.to_dict(),
            "questionnaire": [
                {
                    "question_id": question.question_id,
                    "prompt": question.prompt,
                }
                for question in record.perturbation_questions
            ],
        }
        for record in records
    ]


def build_persona_stage_packet(
    records: Sequence[SampleAuditRecord],
    *,
    dimension: Literal["trust", "usability", "understandability"],
    top_attributions: int = 3,
) -> str:
    sections: list[str] = []
    for record in records:
        model_output = f"label={record.prediction.label}, score={record.prediction.score:.3f}"
        explanation_line = (
            record.explanation.explanation_text
            if record.explanation.explanation_text
            else "No natural-language explanation text provided."
        )

        lines = [
            f"{record.sample_id}",
            f"instance: {_format_instance(record.instance)}",
            f"model_output: {model_output}",
            f"explanation: {explanation_line}",
        ]

        if dimension == "understandability":
            for question in record.perturbation_questions:
                lines.append(f"question {question.question_id}: {question.prompt}")

        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def _score_direction(delta: float) -> str:
    if abs(delta) < 1e-6:
        return "no_change"
    return "increase" if delta > 0 else "decrease"


def _format_instance(instance: TabularRow) -> str:
    return ", ".join(f"{key}={_format_scalar(value)}" for key, value in instance.items())


def _format_scalar(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _format_top_attributions(record: SampleAuditRecord, limit: int) -> str:
    ranked = sorted(
        record.explanation.feature_attributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:limit]
    if not ranked:
        return "none"
    return ", ".join(f"{feature}={value:.3f}" for feature, value in ranked)


def _split_instance_and_target(
    row: TabularRow,
    model: TabularBlackBoxModel,
) -> tuple[TabularRow, str | None]:
    feature_names = set(getattr(model, "feature_names", []))
    instance = {key: value for key, value in row.items() if not feature_names or key in feature_names}

    target_value = None
    for target_key in ("target", "label", "ground_truth_label", "y_true"):
        if target_key in row:
            target_value = row[target_key]
            break

    return instance, _normalize_ground_truth_label(model, target_value)


def _normalize_ground_truth_label(
    model: TabularBlackBoxModel,
    raw_label: object,
) -> str | None:
    if raw_label is None:
        return None

    classes = tuple(str(label) for label in getattr(model, "classes_", ()))
    if isinstance(raw_label, str) and raw_label in classes:
        return raw_label

    if isinstance(raw_label, bool):
        index = int(raw_label)
        if 0 <= index < len(classes):
            return classes[index]

    if isinstance(raw_label, int) and not isinstance(raw_label, bool):
        if 0 <= raw_label < len(classes):
            return classes[raw_label]

    if isinstance(raw_label, float) and raw_label.is_integer():
        index = int(raw_label)
        if 0 <= index < len(classes):
            return classes[index]

    normalized = str(raw_label)
    if normalized.isdigit():
        index = int(normalized)
        if 0 <= index < len(classes):
            return classes[index]
    return normalized
