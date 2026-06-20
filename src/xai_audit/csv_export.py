from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Any

from .types import AuditCsvArtifact, AuditCsvRow, PersonaReview, SampleAuditRecord


def build_csv_artifact(
    *,
    interaction_records: list[dict[str, Any]],
    persona_reviews: list[PersonaReview],
    aggregate_scores: dict[str, float],
    sampled_cases: list[SampleAuditRecord],
    csv_path: str | None = None,
) -> AuditCsvArtifact:
    rows = _build_csv_rows(
        persona_reviews=persona_reviews,
        aggregate_scores=aggregate_scores,
    )
    csv_content = _render_csv(rows)

    resolved_path: str | None = None
    if csv_path:
        output_path = Path(csv_path)
        output_path.write_text(csv_content, encoding="utf-8")
        resolved_path = str(output_path.resolve())

    return AuditCsvArtifact(
        csv_content=csv_content,
        csv_rows=rows,
        aggregate_scores=aggregate_scores,
        persona_summaries=persona_reviews,
        sampled_cases=sampled_cases,
        csv_path=resolved_path,
    )


def _build_csv_rows(
    *,
    persona_reviews: list[PersonaReview],
    aggregate_scores: dict[str, float],
) -> list[AuditCsvRow]:
    rows: list[AuditCsvRow] = []

    for review in persona_reviews:
        rows.append(
            AuditCsvRow(
                row_type="stage",
                persona=review.persona,
                stakeholder_role=review.stakeholder_role,
                stage="trust",
                aspect="trust",
                score=review.trust_score,
                summary=review.trust_rationale,
                accuracy=_nested_accuracy(review.scoring_breakdown, "trust"),
            )
        )
        for answer in review.trust_answers:
            rows.append(
                AuditCsvRow(
                    row_type="interaction",
                    persona=review.persona,
                    stakeholder_role=review.stakeholder_role,
                    stage="trust_question",
                    aspect="trust",
                    question_id=answer.question_id,
                    sample_id=_sample_id_from_question(answer.question_id),
                    perceived_correct=str(answer.perceived_correct),
                    actual_correct=str(answer.actual_correct),
                    is_alignment_correct=str(answer.is_alignment_correct),
                )
            )

        rows.append(
            AuditCsvRow(
                row_type="stage",
                persona=review.persona,
                stakeholder_role=review.stakeholder_role,
                stage="usability",
                aspect="usability",
                score=review.usability_score,
                summary=review.usability_rationale,
                accuracy=_nested_accuracy(review.scoring_breakdown, "usability"),
            )
        )
        for rating in review.usability_ratings:
            rows.append(
                AuditCsvRow(
                    row_type="interaction",
                    persona=review.persona,
                    stakeholder_role=review.stakeholder_role,
                    stage="usability_question",
                    aspect="usability",
                    question_id=rating.question_id,
                    sample_id=_sample_id_from_question(rating.question_id),
                    usability_rating=float(rating.usability_score),
                )
            )

        rows.append(
            AuditCsvRow(
                row_type="stage",
                persona=review.persona,
                stakeholder_role=review.stakeholder_role,
                stage="understandability",
                aspect="understandability",
                score=review.understandability_score,
                summary=review.understandability_summary,
                accuracy=_nested_accuracy(review.scoring_breakdown, "understandability"),
            )
        )
        for answer in review.question_answers:
            rows.append(
                AuditCsvRow(
                    row_type="interaction",
                    persona=review.persona,
                    stakeholder_role=review.stakeholder_role,
                    stage="understandability_question",
                    aspect="understandability",
                    question_id=answer.question_id,
                    sample_id=_sample_id_from_question(answer.question_id),
                    predicted_label=answer.predicted_label,
                    predicted_score_direction=answer.predicted_score_direction,
                    is_label_correct=str(answer.is_label_correct),
                    is_direction_correct=str(answer.is_direction_correct),
                )
            )

    rows.extend(
        [
            AuditCsvRow(
                row_type="overall",
                persona="ALL",
                stakeholder_role="ALL",
                stage="overall",
                aspect="trust",
                score=aggregate_scores.get("trust_score_avg"),
            ),
            AuditCsvRow(
                row_type="overall",
                persona="ALL",
                stakeholder_role="ALL",
                stage="overall",
                aspect="usability",
                score=aggregate_scores.get("usability_score_avg"),
            ),
            AuditCsvRow(
                row_type="overall",
                persona="ALL",
                stakeholder_role="ALL",
                stage="overall",
                aspect="understandability",
                score=aggregate_scores.get("understandability_score_avg"),
            ),
        ]
    )
    return rows


def _render_csv(rows: list[AuditCsvRow]) -> str:
    buffer = StringIO()
    fieldnames = list(AuditCsvRow.__dataclass_fields__.keys())
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row.to_dict())
    return buffer.getvalue()


def _sample_id_from_question(question_id: str) -> str:
    if question_id.endswith("_trust"):
        return question_id[:-6]
    if question_id.endswith("_usability"):
        return question_id[:-10]
    parts = question_id.split("_q", maxsplit=1)
    return parts[0] if parts else ""


def _nested_accuracy(scoring_breakdown: dict[str, Any], key: str) -> float | None:
    section = scoring_breakdown.get(key, {})
    if "accuracy" in section:
        return float(section["accuracy"])
    if "average" in section:
        return float(section["average"])
    return None
