from __future__ import annotations

from collections import defaultdict
from typing import Any

from .types import (
    PersonaEvaluationInteraction,
    PersonaQuestionAnswer,
    PersonaReview,
    PersonaTrustAnswer,
    PersonaUsabilityRating,
)


def compose_persona_reviews(
    interaction_records: list[dict[str, Any]],
    sampled_cases: list[dict[str, Any]],
) -> list[PersonaReview]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    ordered_interactions: dict[str, list[PersonaEvaluationInteraction]] = defaultdict(list)

    for record in interaction_records:
        persona = record["persona"]
        dimension = record["dimension"]
        grouped[persona][dimension].append(record["persona_response"])
        ordered_interactions[persona].append(
            PersonaEvaluationInteraction(
                persona=persona,
                stakeholder_role=record.get("stakeholder_role", ""),
                dimension=dimension,
                coordinator_prompt=record["coordinator_prompt"],
                persona_response=record["persona_response"],
            )
        )

    reviews: list[PersonaReview] = []
    for persona in sorted(grouped):
        trust_result = _grade_trust(grouped[persona].get("trust", []), sampled_cases)
        usability_result = _grade_usability(grouped[persona].get("usability", []))
        understandability_result = _grade_understandability(
            grouped[persona].get("understandability", []),
            sampled_cases,
        )
        stakeholder_role = ordered_interactions[persona][0].stakeholder_role if ordered_interactions[persona] else ""

        reviews.append(
            PersonaReview(
                persona=persona,
                stakeholder_role=stakeholder_role,
                trust_score=trust_result["trust_score"],
                trust_rationale=trust_result["trust_rationale"],
                usability_score=usability_result["usability_score"],
                usability_rationale=usability_result["usability_rationale"],
                understandability_score=understandability_result["understandability_score"],
                understandability_summary=understandability_result["understandability_summary"],
                strengths=[],
                concerns=[],
                trust_answers=[
                    PersonaTrustAnswer(**answer)
                    for answer in trust_result["trust_answers"]
                ],
                usability_ratings=[
                    PersonaUsabilityRating(**rating)
                    for rating in usability_result["usability_ratings"]
                ],
                question_answers=[
                    PersonaQuestionAnswer(**answer)
                    for answer in understandability_result["question_answers"]
                ],
                scoring_breakdown={
                    "trust": trust_result["scoring_breakdown"],
                    "usability": usability_result["scoring_breakdown"],
                    "understandability": understandability_result["scoring_breakdown"],
                },
                interaction_history=ordered_interactions[persona],
            )
        )

    return reviews


def aggregate_scores(persona_reviews: list[PersonaReview]) -> dict[str, float]:
    if not persona_reviews:
        return {
            "trust_score_avg": 0.0,
            "usability_score_avg": 0.0,
            "understandability_score_avg": 0.0,
        }

    count = len(persona_reviews)
    return {
        "trust_score_avg": round(sum(review.trust_score for review in persona_reviews) / count, 4),
        "usability_score_avg": round(sum(review.usability_score for review in persona_reviews) / count, 4),
        "understandability_score_avg": round(
            sum(review.understandability_score for review in persona_reviews) / count,
            4,
        ),
    }


def _grade_trust(
    trust_records: list[dict[str, Any]],
    sampled_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    correctness_key = {
        f"{record['sample_id']}_trust": bool(record["prediction_is_correct"])
        for record in sampled_cases
        if record.get("prediction_is_correct") is not None
    }
    graded_answers: list[dict[str, Any]] = []
    aligned = 0
    total = 0

    for answer in trust_records:
        question_id = str(answer.get("question_id", "")).strip()
        actual_correct = correctness_key.get(question_id)
        if actual_correct is None:
            continue
        perceived_correct = bool(answer.get("perceived_correct"))
        is_alignment_correct = perceived_correct == actual_correct
        graded_answers.append(
            PersonaTrustAnswer(
                question_id=question_id,
                perceived_correct=perceived_correct,
                actual_correct=actual_correct,
                is_alignment_correct=is_alignment_correct,
            ).to_dict()
        )
        aligned += int(is_alignment_correct)
        total += 1

    trust_score = (aligned / total) if total else 0.0
    return {
        "trust_score": round(trust_score, 4),
        "trust_rationale": f"Perceived correctness aligned with actual correctness on {aligned}/{total} samples." if total else "No trust questions were graded.",
        "trust_answers": graded_answers,
        "scoring_breakdown": {
            "questions_graded": total,
            "aligned_judgments": aligned,
            "accuracy": round(trust_score, 4),
        },
    }


def _grade_usability(usability_records: list[dict[str, Any]]) -> dict[str, Any]:
    ratings: list[dict[str, Any]] = []
    numeric_scores: list[int] = []

    for answer in usability_records:
        question_id = str(answer.get("question_id", "")).strip()
        try:
            usability_score = max(1, min(5, int(answer.get("usability_score", 1))))
        except (TypeError, ValueError):
            usability_score = 1
        ratings.append(
            PersonaUsabilityRating(
                question_id=question_id,
                usability_score=usability_score,
            ).to_dict()
        )
        numeric_scores.append(usability_score)

    average_score = (sum(numeric_scores) / len(numeric_scores)) if numeric_scores else 0.0
    return {
        "usability_score": round(average_score, 4),
        "usability_rationale": f"Average usability score was {average_score:.2f} across {len(numeric_scores)} samples." if numeric_scores else "No usability questions were graded.",
        "usability_ratings": ratings,
        "scoring_breakdown": {
            "questions_graded": len(numeric_scores),
            "score_sum": sum(numeric_scores),
            "average": round(average_score, 4),
        },
    }


def _grade_understandability(
    understandability_records: list[dict[str, Any]],
    sampled_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    answer_key = {
        question["question_id"]: question
        for record in sampled_cases
        for question in record.get("perturbation_questions", [])
    }
    graded_answers: list[dict[str, Any]] = []
    points_earned = 0
    points_possible = 0

    for answer in understandability_records:
        question_id = str(answer.get("question_id", "")).strip()
        expected = answer_key.get(question_id)
        if expected is None:
            continue
        predicted_label = str(answer.get("predicted_label", "")).strip()
        predicted_direction = str(answer.get("predicted_score_direction", "")).strip()
        is_label_correct = predicted_label.lower() == str(expected["expected_label"]).strip().lower()
        is_direction_correct = predicted_direction == expected["expected_score_direction"]
        graded_answers.append(
            PersonaQuestionAnswer(
                question_id=question_id,
                predicted_label=predicted_label,
                predicted_score_direction=predicted_direction,  # type: ignore[arg-type]
                is_label_correct=is_label_correct,
                is_direction_correct=is_direction_correct,
            ).to_dict()
        )
        points_earned += int(is_label_correct) + int(is_direction_correct)
        points_possible += 2

    accuracy = (points_earned / points_possible) if points_possible else 0.0
    return {
        "understandability_score": round(accuracy, 4),
        "understandability_summary": f"Perturbation-question accuracy was {accuracy:.0%} across {points_possible // 2} samples." if points_possible else "No understandability questions were graded.",
        "question_answers": graded_answers,
        "scoring_breakdown": {
            "questions_graded": points_possible // 2,
            "points_earned": points_earned,
            "points_possible": points_possible,
            "accuracy": round(accuracy, 4),
        },
    }
