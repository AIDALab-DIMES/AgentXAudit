from __future__ import annotations

import json
import operator
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Annotated, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from .interfaces import FixedTabularExplainer, TabularBlackBoxModel
from .questionnaire import ValidationSamplingConfig, build_sample_audit_records, user_visible_records
from .types import AuditReport, PersonaQuestionAnswer, PersonaReview, PredictionResult, TabularRow


class AuditState(TypedDict, total=False):
    user_query: str
    validation_set: list[TabularRow]
    reference_row: TabularRow | None
    focus_label: str | None
    sample_count: int
    sampled_cases: list[dict[str, Any]]
    persona_reviews: Annotated[list[dict[str, Any]], operator.add]
    final_report: dict[str, Any]


@dataclass
class AuditRuntimeContext:
    llm: BaseChatModel
    model: TabularBlackBoxModel
    explainer: FixedTabularExplainer


@dataclass
class PersonaQuestionAnswerSchema:
    question_id: str
    predicted_label: str
    predicted_score_direction: Literal["increase", "decrease", "no_change"]
    reasoning: str


@dataclass
class PersonaQuestionnaireSchema:
    persona: str
    trust_score: int
    trust_rationale: str
    usability_score: int
    usability_rationale: str
    understanding_observations: str
    strengths: list[str]
    concerns: list[str]
    question_answers: list[PersonaQuestionAnswerSchema]


@dataclass
class FinalReportSchema:
    verdict: Literal["pass", "review", "fail"]
    executive_summary: str
    recommendations: list[str]


PERSONA_PROMPTS = {
    "data_scientist": """
You are simulating a senior data scientist auditing a tabular explainability workflow.
Use the sampled validation cases, black-box outputs, and explanation outputs to judge whether the explanation method is credible,
diagnostically useful, and consistent enough to support practitioner trust.
Score trust and usability from 1 to 5, where 1 is very poor and 5 is very strong.
Then answer each perturbation question by predicting the black-box model's likely output.
""".strip(),
    "business_executive": """
You are simulating a business executive reviewing whether a tabular explanation method is decision-ready.
Use the sampled cases to judge whether leaders could rely on these explanations in operations, governance, and communication.
Score trust and usability from 1 to 5, where 1 is very poor and 5 is very strong.
Then answer each perturbation question by predicting the black-box model's likely output.
""".strip(),
    "end_user": """
You are simulating an end user affected by model-based decisions.
Use the sampled cases to judge whether the model outputs and explanations would feel understandable, fair, and actionable.
Score trust and usability from 1 to 5, where 1 is very poor and 5 is very strong.
Then answer each perturbation question by predicting the black-box model's likely output.
""".strip(),
}


def build_audit_graph() -> Any:
    builder = StateGraph(AuditState, context_schema=AuditRuntimeContext)
    builder.add_node("initialize_audit", _initialize_audit)
    builder.add_node("data_scientist_review", _persona_node("data_scientist"))
    builder.add_node("business_executive_review", _persona_node("business_executive"))
    builder.add_node("end_user_review", _persona_node("end_user"))
    builder.add_node("synthesize_report", _synthesize_report)

    builder.add_edge(START, "initialize_audit")
    builder.add_edge("initialize_audit", "data_scientist_review")
    builder.add_edge("initialize_audit", "business_executive_review")
    builder.add_edge("initialize_audit", "end_user_review")
    builder.add_edge(
        [
            "data_scientist_review",
            "business_executive_review",
            "end_user_review",
        ],
        "synthesize_report",
    )
    builder.add_edge("synthesize_report", END)
    return builder.compile()


class AuditCoordinator:
    def __init__(
        self,
        *,
        llm: BaseChatModel,
        sample_count: int = 5,
    ) -> None:
        self._graph = build_audit_graph()
        self._llm = llm
        self._sample_count = sample_count

    def audit(
        self,
        *,
        user_query: str,
        validation_set: list[TabularRow],
        model: TabularBlackBoxModel,
        explainer: FixedTabularExplainer,
        reference_row: TabularRow | None = None,
        focus_label: str | None = None,
        sample_count: int | None = None,
    ) -> dict[str, Any]:
        context = AuditRuntimeContext(
            llm=self._llm,
            model=model,
            explainer=explainer,
        )
        initial_state: AuditState = {
            "user_query": user_query,
            "validation_set": validation_set,
            "reference_row": reference_row,
            "focus_label": focus_label,
            "sample_count": sample_count or self._sample_count,
            "persona_reviews": [],
        }
        return self._graph.invoke(initial_state, context=context)


def _initialize_audit(state: AuditState, runtime: Runtime[AuditRuntimeContext]) -> dict[str, Any]:
    records = build_sample_audit_records(
        model=runtime.context.model,
        explainer=runtime.context.explainer,
        validation_set=state["validation_set"],
        config=ValidationSamplingConfig(sample_count=state["sample_count"]),
        reference_row=state.get("reference_row"),
        focus_label=state.get("focus_label"),
    )
    return {"sampled_cases": [record.to_dict() for record in records]}


def _persona_node(persona_key: str):
    persona_name = persona_key.replace("_", " ")

    def run_persona(state: AuditState, runtime: Runtime[AuditRuntimeContext]) -> dict[str, Any]:
        llm = runtime.context.llm.with_structured_output(PersonaQuestionnaireSchema)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PERSONA_PROMPTS[persona_key]),
                (
                    "human",
                    """
Audit the explanation method using the following sampled validation cases.

User query:
{user_query}

Sampled cases with black-box outputs and explanation outputs:
{sampled_cases}

Instructions:
- Give a trust score from 1 to 5.
- Give a usability score from 1 to 5.
- Answer every perturbation question.
- For `predicted_score_direction`, only use: increase, decrease, no_change.
- Keep your strengths and concerns tied to evidence in the sampled cases.
                    """.strip(),
                ),
            ]
        )
        chain = prompt | llm
        visible_cases = user_visible_records(_dicts_to_sample_records(state["sampled_cases"]))
        review = chain.invoke(
            {
                "user_query": state["user_query"],
                "sampled_cases": json.dumps(visible_cases, indent=2, sort_keys=True),
            }
        )
        normalized = _normalize_result(review)
        normalized["persona"] = normalized.get("persona") or persona_name
        graded_review = _grade_persona_review(normalized, state["sampled_cases"])
        return {"persona_reviews": [graded_review]}

    return run_persona


def _synthesize_report(state: AuditState, runtime: Runtime[AuditRuntimeContext]) -> dict[str, Any]:
    llm = runtime.context.llm.with_structured_output(FinalReportSchema)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are the audit coordinator synthesizing a final report about a tabular explanation method.
You are given validation-set evidence, black-box model outputs, explanation outputs, and persona questionnaire results.
Use these to produce a concise decision-oriented synthesis.
                """.strip(),
            ),
            (
                "human",
                """
Prepare the final audit report.

User query:
{user_query}

Aggregate scores:
{aggregate_scores}

Persona reviews:
{persona_reviews}

Return a verdict, an executive summary, and actionable recommendations.
                """.strip(),
            ),
        ]
    )
    aggregate_scores = _aggregate_scores(state.get("persona_reviews", []))
    chain = prompt | llm
    report = chain.invoke(
        {
            "user_query": state["user_query"],
            "aggregate_scores": json.dumps(aggregate_scores, indent=2, sort_keys=True),
            "persona_reviews": json.dumps(state.get("persona_reviews", []), indent=2, sort_keys=True),
        }
    )
    normalized = _normalize_result(report)
    final_report = AuditReport(
        verdict=normalized["verdict"],
        executive_summary=normalized["executive_summary"],
        aggregate_scores=aggregate_scores,
        sampled_cases=_dicts_to_sample_records(state.get("sampled_cases", [])),
        persona_summaries=[_dict_to_persona_review(review) for review in state.get("persona_reviews", [])],
        recommendations=normalized.get("recommendations", []),
    )
    return {"final_report": final_report.to_dict()}


def _grade_persona_review(review: dict[str, Any], sampled_cases: list[dict[str, Any]]) -> dict[str, Any]:
    answer_key = {
        question["question_id"]: question
        for record in sampled_cases
        for question in record.get("perturbation_questions", [])
    }
    graded_answers: list[dict[str, Any]] = []
    points_earned = 0
    points_possible = 0

    for answer in review.get("question_answers", []):
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
    understanding_score = _accuracy_to_five_point_scale(accuracy)

    review["trust_score"] = _clamp_five_point_score(review.get("trust_score", 1))
    review["usability_score"] = _clamp_five_point_score(review.get("usability_score", 1))
    review["understanding_score"] = understanding_score
    review["understanding_summary"] = (
        f"{review.get('understanding_observations', '').strip()} "
        f"Quiz accuracy was {accuracy:.0%} across {points_possible // 2} perturbation questions."
    ).strip()
    review["question_answers"] = graded_answers
    review["scoring_breakdown"] = {
        "questions_graded": points_possible // 2,
        "points_earned": points_earned,
        "points_possible": points_possible,
        "accuracy": round(accuracy, 4),
    }
    review.pop("understanding_observations", None)
    return review


def _aggregate_scores(persona_reviews: list[dict[str, Any]]) -> dict[str, float]:
    if not persona_reviews:
        return {
            "trust_score_avg": 0.0,
            "usability_score_avg": 0.0,
            "understanding_score_avg": 0.0,
        }

    count = len(persona_reviews)
    return {
        "trust_score_avg": round(sum(review["trust_score"] for review in persona_reviews) / count, 2),
        "usability_score_avg": round(sum(review["usability_score"] for review in persona_reviews) / count, 2),
        "understanding_score_avg": round(sum(review["understanding_score"] for review in persona_reviews) / count, 2),
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


def _normalize_result(result: Any) -> dict[str, Any]:
    if is_dataclass(result):
        return asdict(result)
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if isinstance(result, dict):
        return result
    raise TypeError(f"Unsupported structured output type: {type(result)!r}")


def _dicts_to_sample_records(payload: list[dict[str, Any]]):
    from .types import LocalExplanation, PerturbationQuestion, SampleAuditRecord

    records: list[SampleAuditRecord] = []
    for item in payload:
        prediction = PredictionResult(**item["prediction"])
        explanation_payload = item["explanation"]
        explanation = LocalExplanation(
            method_name=explanation_payload["method_name"],
            prediction=PredictionResult(**explanation_payload["prediction"]),
            feature_attributions=explanation_payload["feature_attributions"],
            baseline_value=explanation_payload.get("baseline_value"),
            explanation_text=explanation_payload.get("explanation_text", ""),
            metadata=explanation_payload.get("metadata", {}),
        )
        questions = [PerturbationQuestion(**question) for question in item.get("perturbation_questions", [])]
        records.append(
            SampleAuditRecord(
                sample_id=item["sample_id"],
                instance=item["instance"],
                prediction=prediction,
                explanation=explanation,
                perturbation_questions=questions,
            )
        )
    return records


def _dict_to_persona_review(payload: dict[str, Any]) -> PersonaReview:
    return PersonaReview(
        persona=payload["persona"],
        trust_score=payload["trust_score"],
        trust_rationale=payload["trust_rationale"],
        usability_score=payload["usability_score"],
        usability_rationale=payload["usability_rationale"],
        understanding_score=payload["understanding_score"],
        understanding_summary=payload["understanding_summary"],
        strengths=payload.get("strengths", []),
        concerns=payload.get("concerns", []),
        question_answers=[PersonaQuestionAnswer(**answer) for answer in payload.get("question_answers", [])],
        scoring_breakdown=payload.get("scoring_breakdown", {}),
    )
