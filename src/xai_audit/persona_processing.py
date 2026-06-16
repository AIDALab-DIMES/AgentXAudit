from __future__ import annotations

from collections import defaultdict
from typing import Any

from .types import PersonaEvaluationInteraction, PersonaQuestionAnswer, PersonaReview


def compose_persona_reviews(
    interaction_records: list[dict[str, Any]],
    sampled_cases: list[dict[str, Any]],
) -> list[PersonaReview]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    ordered_interactions: dict[str, list[PersonaEvaluationInteraction]] = defaultdict(list)

    for record in interaction_records:
        persona = record["persona"]
        dimension = record["dimension"]
        grouped[persona][dimension] = record["persona_response"]
        ordered_interactions[persona].append(
            PersonaEvaluationInteraction(
                persona=persona,
                dimension=dimension,
                coordinator_prompt=record["coordinator_prompt"],
                persona_response=record["persona_response"],
            )
        )

    reviews: list[PersonaReview] = []
    for persona in sorted(grouped):
        trust_data = grouped[persona].get("trust", {})
        usability_data = grouped[persona].get("usability", {})
        understanding_data = grouped[persona].get("understanding", {})
        graded_understanding = _grade_understanding(understanding_data, sampled_cases)

        strengths = _merge_unique_lists(
            trust_data.get("strengths", []),
            usability_data.get("strengths", []),
        )
        concerns = _merge_unique_lists(
            trust_data.get("concerns", []),
            usability_data.get("concerns", []),
        )

        reviews.append(
            PersonaReview(
                persona=persona,
                trust_score=_clamp_five_point_score(trust_data.get("trust_score", 1)),
                trust_rationale=trust_data.get("trust_rationale", ""),
                usability_score=_clamp_five_point_score(usability_data.get("usability_score", 1)),
                usability_rationale=usability_data.get("usability_rationale", ""),
                understanding_score=graded_understanding["understanding_score"],
                understanding_summary=graded_understanding["understanding_summary"],
                strengths=strengths,
                concerns=concerns,
                question_answers=[
                    PersonaQuestionAnswer(**answer)
                    for answer in graded_understanding["question_answers"]
                ],
                scoring_breakdown=graded_understanding["scoring_breakdown"],
                interaction_history=ordered_interactions[persona],
            )
        )

    return reviews


def aggregate_scores(persona_reviews: list[PersonaReview]) -> dict[str, float]:
    if not persona_reviews:
        return {
            "trust_score_avg": 0.0,
            "usability_score_avg": 0.0,
            "understanding_score_avg": 0.0,
        }

    count = len(persona_reviews)
    return {
        "trust_score_avg": round(sum(review.trust_score for review in persona_reviews) / count, 2),
        "usability_score_avg": round(sum(review.usability_score for review in persona_reviews) / count, 2),
        "understanding_score_avg": round(sum(review.understanding_score for review in persona_reviews) / count, 2),
    }


def _grade_understanding(
    understanding_data: dict[str, Any],
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

    for answer in understanding_data.get("question_answers", []):
        expected = answer_key.get(answer["question_id"])
        if expected is None:
            continue
        is_label_correct = str(answer["predicted_label"]).strip().lower() == expected["expected_label"].strip().lower()
        is_direction_correct = answer["predicted_score_direction"] == expected["expected_score_direction"]
        graded_answers.append(
            PersonaQuestionAnswer(
                question_id=answer["question_id"],
                predicted_label=answer["predicted_label"],
                predicted_score_direction=answer["predicted_score_direction"],
                reasoning=answer.get("reasoning", ""),
                is_label_correct=is_label_correct,
                is_direction_correct=is_direction_correct,
            ).to_dict()
        )
        points_earned += int(is_label_correct) + int(is_direction_correct)
        points_possible += 2

    accuracy = (points_earned / points_possible) if points_possible else 0.0
    understanding_summary = (
        f"{understanding_data.get('understanding_observations', '').strip()} "
        f"Quiz accuracy was {accuracy:.0%} across {points_possible // 2} perturbation questions."
    ).strip()

    return {
        "understanding_score": _accuracy_to_five_point_scale(accuracy),
        "understanding_summary": understanding_summary,
        "question_answers": graded_answers,
        "scoring_breakdown": {
            "questions_graded": points_possible // 2,
            "points_earned": points_earned,
            "points_possible": points_possible,
            "accuracy": round(accuracy, 4),
        },
    }


def _accuracy_to_five_point_scale(accuracy: float) -> int:
    if accuracy <= 0.2:
        return 1
    if accuracy <= 0.4:
        return 2
    if accuracy <= 0.6:
        return 3
    if accuracy <= 0.8:
        return 4
    return 5


def _clamp_five_point_score(value: Any) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return 1
    return max(1, min(5, numeric))


def _merge_unique_lists(*groups: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for item in group:
            normalized = item.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                merged.append(normalized)
    return merged
