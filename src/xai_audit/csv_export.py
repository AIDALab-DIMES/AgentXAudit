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
        interaction_records=interaction_records,
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
    interaction_records: list[dict[str, Any]],
    persona_reviews: list[PersonaReview],
    aggregate_scores: dict[str, float],
) -> list[AuditCsvRow]:
    rows: list[AuditCsvRow] = []

    for review in persona_reviews:
        rows.append(
            AuditCsvRow(
                row_type="stage",
                persona=review.persona,
                stage="trust",
                aspect="trust",
                score=float(review.trust_score),
                summary=review.trust_rationale,
                strengths=" | ".join(review.strengths),
                concerns=" | ".join(review.concerns),
            )
        )
        rows.append(
            AuditCsvRow(
                row_type="stage",
                persona=review.persona,
                stage="usability",
                aspect="usability",
                score=float(review.usability_score),
                summary=review.usability_rationale,
                strengths=" | ".join(review.strengths),
                concerns=" | ".join(review.concerns),
            )
        )
        rows.append(
            AuditCsvRow(
                row_type="stage",
                persona=review.persona,
                stage="understanding",
                aspect="understanding",
                score=float(review.understanding_score),
                summary=review.understanding_summary,
                accuracy=_coerce_accuracy(review.scoring_breakdown.get("accuracy")),
            )
        )
        for answer in review.question_answers:
            rows.append(
                AuditCsvRow(
                    row_type="interaction",
                    persona=review.persona,
                    stage="understanding_question",
                    aspect="understanding",
                    question_id=answer.question_id,
                    sample_id=_sample_id_from_question(answer.question_id),
                    predicted_label=answer.predicted_label,
                    predicted_score_direction=answer.predicted_score_direction,
                    is_label_correct=str(answer.is_label_correct),
                    is_direction_correct=str(answer.is_direction_correct),
                )
            )

    for record in interaction_records:
        if record["dimension"] != "understanding":
            continue
        response = record["persona_response"]
        if "understanding_observations" in response:
            rows.append(
                AuditCsvRow(
                    row_type="interaction",
                    persona=record["persona"],
                    stage="understanding_observation",
                    aspect="understanding",
                    summary=response["understanding_observations"],
                )
            )

    rows.extend(
        [
            AuditCsvRow(
                row_type="overall",
                persona="ALL",
                stage="overall",
                aspect="trust",
                score=aggregate_scores.get("trust_score_avg"),
            ),
            AuditCsvRow(
                row_type="overall",
                persona="ALL",
                stage="overall",
                aspect="usability",
                score=aggregate_scores.get("usability_score_avg"),
            ),
            AuditCsvRow(
                row_type="overall",
                persona="ALL",
                stage="overall",
                aspect="understanding",
                score=aggregate_scores.get("understanding_score_avg"),
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
    parts = question_id.split("_q", maxsplit=1)
    return parts[0] if parts else ""


def _coerce_accuracy(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
