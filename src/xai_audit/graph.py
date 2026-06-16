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
from .persona_processing import aggregate_scores, compose_persona_reviews
from .questionnaire import (
    ValidationSamplingConfig,
    build_persona_stage_packet,
    build_sample_audit_records,
)
from .types import AuditReport, PersonaReview, TabularRow


class AuditState(TypedDict, total=False):
    user_query: str
    validation_set: list[TabularRow]
    focus_label: str | None
    sample_count: int
    sampled_cases: list[dict[str, Any]]
    interaction_records: Annotated[list[dict[str, Any]], operator.add]
    final_report: dict[str, Any]


@dataclass
class AuditRuntimeContext:
    llm: BaseChatModel
    model: TabularBlackBoxModel
    explainer: FixedTabularExplainer


@dataclass
class TrustEvaluationSchema:
    persona: str
    trust_score: int
    trust_rationale: str
    strengths: list[str]
    concerns: list[str]


@dataclass
class UsabilityEvaluationSchema:
    persona: str
    usability_score: int
    usability_rationale: str
    strengths: list[str]
    concerns: list[str]


@dataclass
class PersonaQuestionAnswerSchema:
    question_id: str
    predicted_label: str
    predicted_score_direction: Literal["increase", "decrease", "no_change"]
    reasoning: str


@dataclass
class UnderstandingEvaluationSchema:
    persona: str
    understanding_observations: str
    question_answers: list[PersonaQuestionAnswerSchema]


@dataclass
class FinalReportSchema:
    verdict: Literal["pass", "review", "fail"]
    executive_summary: str
    recommendations: list[str]


PERSONA_PROMPTS = {
    "data_scientist": """
You are simulating a senior data scientist auditing a tabular explainability workflow.
Ground all judgments in the sampled validation cases, black-box outputs, and explanation outputs.
Be concrete, evidence-based, and appropriately critical.
""".strip(),
    "business_executive": """
You are simulating a business executive reviewing whether a tabular explanation method is decision-ready.
Ground all judgments in the sampled validation cases, black-box outputs, and explanation outputs.
Be concise, practical, and risk-aware.
""".strip(),
    "end_user": """
You are simulating an end user affected by model-based decisions.
Ground all judgments in the sampled validation cases, black-box outputs, and explanation outputs.
Focus on clarity, fairness, and actionability.
""".strip(),
}

DIMENSION_INSTRUCTIONS = {
    "trust": """
This interaction evaluates trust only.
Score trust from 1 to 5, where 1 means very low trust and 5 means very high trust.
Do not evaluate usability or model understanding here.
Return evidence-based strengths and concerns related only to trust.
""".strip(),
    "usability": """
This interaction evaluates usability only.
Score usability from 1 to 5, where 1 means very poor usability and 5 means very strong usability.
Do not evaluate trust or model understanding here.
Return evidence-based strengths and concerns related only to usability.
""".strip(),
    "understanding": """
This interaction evaluates model understanding only.
Do not score trust or usability here.
Answer every perturbation question by predicting the black-box model's likely label and whether the focal score would increase, decrease, or stay about the same.
Use only these values for predicted_score_direction: increase, decrease, no_change.
Also provide a short observation about how understandable the model behavior felt from the sampled evidence.
""".strip(),
}

PERSONA_ORDER = ("data_scientist", "business_executive", "end_user")


def build_audit_graph() -> Any:
    builder = StateGraph(AuditState, context_schema=AuditRuntimeContext)
    builder.add_node("initialize_audit", _initialize_audit)

    terminal_nodes: list[str] = []
    for persona_key in PERSONA_ORDER:
        trust_node = f"{persona_key}_trust_review"
        usability_node = f"{persona_key}_usability_review"
        understanding_node = f"{persona_key}_understanding_review"

        builder.add_node(trust_node, _stage_node(persona_key, "trust", TrustEvaluationSchema))
        builder.add_node(usability_node, _stage_node(persona_key, "usability", UsabilityEvaluationSchema))
        builder.add_node(understanding_node, _stage_node(persona_key, "understanding", UnderstandingEvaluationSchema))

        builder.add_edge("initialize_audit", trust_node)
        builder.add_edge(trust_node, usability_node)
        builder.add_edge(usability_node, understanding_node)
        terminal_nodes.append(understanding_node)

    builder.add_node("synthesize_report", _synthesize_report)
    builder.add_edge(START, "initialize_audit")
    builder.add_edge(terminal_nodes, "synthesize_report")
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
            "interaction_records": [],
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


def _stage_node(persona_key: str, dimension: str, schema: type[Any]):
    persona_name = persona_key.replace("_", " ")

    def run_stage(state: AuditState, runtime: Runtime[AuditRuntimeContext]) -> dict[str, Any]:
        llm = runtime.context.llm.with_structured_output(schema)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PERSONA_PROMPTS[persona_key]),
                (
                    "human",
                    """
You are interacting with the audit coordinator in one stage of a multi-step evaluation.

Coordinator request:
{coordinator_prompt}
                    """.strip(),
                ),
            ]
        )
        prompt_packet = build_persona_stage_packet(
            _dicts_to_sample_records(state["sampled_cases"]),
            dimension=dimension,
        )
        coordinator_prompt = _build_coordinator_prompt(
            dimension=dimension,
            user_query=state["user_query"],
            sampled_cases_text=prompt_packet,
        )
        chain = prompt | llm
        response = chain.invoke({"coordinator_prompt": coordinator_prompt})
        normalized = _normalize_result(response)
        normalized["persona"] = normalized.get("persona") or persona_name
        interaction_record = {
            "persona": normalized["persona"],
            "dimension": dimension,
            "coordinator_prompt": coordinator_prompt,
            "persona_response": normalized,
        }
        return {"interaction_records": [interaction_record]}

    return run_stage


def _synthesize_report(state: AuditState, runtime: Runtime[AuditRuntimeContext]) -> dict[str, Any]:
    persona_reviews = compose_persona_reviews(
        state.get("interaction_records", []),
        state.get("sampled_cases", []),
    )
    aggregate = aggregate_scores(persona_reviews)
    persona_review_summaries = [_persona_review_summary(review) for review in persona_reviews]

    llm = runtime.context.llm.with_structured_output(FinalReportSchema)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are the audit coordinator synthesizing a final report about a tabular explanation method.
The persona evaluations were performed iteratively in separate trust, usability, and understanding interactions over the same sampled validation cases.
Use this staged evidence to produce a concise decision-oriented synthesis.
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
    chain = prompt | llm
    report = chain.invoke(
        {
            "user_query": state["user_query"],
            "aggregate_scores": json.dumps(aggregate, indent=2, sort_keys=True),
            "persona_reviews": json.dumps(persona_review_summaries, indent=2, sort_keys=True),
        }
    )
    normalized = _normalize_result(report)
    final_report = AuditReport(
        verdict=normalized["verdict"],
        executive_summary=normalized["executive_summary"],
        aggregate_scores=aggregate,
        sampled_cases=_dicts_to_sample_records(state.get("sampled_cases", [])),
        persona_summaries=persona_reviews,
        recommendations=normalized.get("recommendations", []),
    )
    return {"final_report": final_report.to_dict()}


def _build_coordinator_prompt(
    *,
    dimension: str,
    user_query: str,
    sampled_cases_text: str,
) -> str:
    return (
        f"User query:\n{user_query}\n\n"
        f"Evaluation dimension:\n{dimension}\n\n"
        f"Stage instructions:\n{DIMENSION_INSTRUCTIONS[dimension]}\n\n"
        "Use the exact same sampled validation cases used in the other stages of this audit.\n\n"
        f"Sampled cases with black-box outputs and explanation outputs:\n{sampled_cases_text}"
    )


def _normalize_result(result: Any) -> dict[str, Any]:
    if is_dataclass(result):
        return asdict(result)
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if isinstance(result, dict):
        return result
    raise TypeError(f"Unsupported structured output type: {type(result)!r}")


def _persona_review_summary(review: PersonaReview) -> dict[str, Any]:
    return {
        "persona": review.persona,
        "trust_score": review.trust_score,
        "trust_rationale": review.trust_rationale,
        "usability_score": review.usability_score,
        "usability_rationale": review.usability_rationale,
        "understanding_score": review.understanding_score,
        "understanding_summary": review.understanding_summary,
        "strengths": review.strengths,
        "concerns": review.concerns,
        "scoring_breakdown": review.scoring_breakdown,
    }


def _dicts_to_sample_records(payload: list[dict[str, Any]]):
    from .types import LocalExplanation, PerturbationQuestion, PredictionResult, SampleAuditRecord

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
