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
from .technical import TechnicalAssessor, TechnicalAssessmentConfig
from .types import AuditReport, PersonaReview, TabularRow


class AuditState(TypedDict, total=False):
    user_query: str
    instance: TabularRow
    reference_row: TabularRow | None
    focus_label: str | None
    explanation_packet: dict[str, Any]
    persona_reviews: Annotated[list[dict[str, Any]], operator.add]
    technical_assessment: dict[str, Any]
    final_report: dict[str, Any]


@dataclass
class AuditRuntimeContext:
    llm: BaseChatModel
    model: TabularBlackBoxModel
    explainer: FixedTabularExplainer
    technical_assessor: TechnicalAssessor


@dataclass
class PersonaReviewSchema:
    persona: str
    summary: str
    strengths: list[str]
    concerns: list[str]
    follow_up_questions: list[str]
    trust_level: Literal["low", "medium", "high"]


@dataclass
class FinalReportSchema:
    verdict: Literal["pass", "review", "fail"]
    executive_summary: str
    user_risks: list[str]
    recommendations: list[str]


PERSONA_PROMPTS = {
    "data_scientist": """
You are simulating a senior data scientist auditing one local explanation for a tabular prediction system.
Focus on feature attribution plausibility, hidden assumptions, risk of spurious correlations, feature interactions,
and whether the explanation would help an ML practitioner debug or challenge the model behavior.
Be appropriately critical and specific.
""".strip(),
    "business_executive": """
You are simulating a business executive reviewing whether a model explanation is decision-useful.
Focus on strategic clarity, decision accountability, operational actionability, ambiguity, and business risk.
Be concise, concrete, and non-technical when possible.
""".strip(),
    "end_user": """
You are simulating an end user affected by the decision.
Focus on clarity, fairness, jargon, emotional impact, contestability, and whether the explanation tells the user what
they can understand or do next. Highlight confusing or patronizing phrasing.
""".strip(),
}


def build_audit_graph() -> Any:
    builder = StateGraph(AuditState, context_schema=AuditRuntimeContext)
    builder.add_node("initialize_audit", _initialize_audit)
    builder.add_node("data_scientist_review", _persona_node("data_scientist"))
    builder.add_node("business_executive_review", _persona_node("business_executive"))
    builder.add_node("end_user_review", _persona_node("end_user"))
    builder.add_node("technical_assessment", _technical_assessment)
    builder.add_node("synthesize_report", _synthesize_report)

    builder.add_edge(START, "initialize_audit")
    builder.add_edge("initialize_audit", "data_scientist_review")
    builder.add_edge("initialize_audit", "business_executive_review")
    builder.add_edge("initialize_audit", "end_user_review")
    builder.add_edge("initialize_audit", "technical_assessment")
    builder.add_edge(
        [
            "data_scientist_review",
            "business_executive_review",
            "end_user_review",
            "technical_assessment",
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
        technical_config: TechnicalAssessmentConfig | None = None,
    ) -> None:
        self._graph = build_audit_graph()
        self._llm = llm
        self._technical_assessor = TechnicalAssessor(technical_config)

    def audit(
        self,
        *,
        user_query: str,
        instance: TabularRow,
        model: TabularBlackBoxModel,
        explainer: FixedTabularExplainer,
        reference_row: TabularRow | None = None,
        focus_label: str | None = None,
    ) -> dict[str, Any]:
        context = AuditRuntimeContext(
            llm=self._llm,
            model=model,
            explainer=explainer,
            technical_assessor=self._technical_assessor,
        )
        initial_state: AuditState = {
            "user_query": user_query,
            "instance": instance,
            "reference_row": reference_row,
            "focus_label": focus_label,
            "persona_reviews": [],
        }
        return self._graph.invoke(initial_state, context=context)


def _initialize_audit(state: AuditState, runtime: Runtime[AuditRuntimeContext]) -> dict[str, Any]:
    explanation = runtime.context.explainer.explain_instance(
        runtime.context.model,
        state["instance"],
        reference_row=state.get("reference_row"),
        focus_label=state.get("focus_label"),
    )
    return {"explanation_packet": explanation.to_dict()}


def _persona_node(persona_key: str):
    persona_name = persona_key.replace("_", " ")

    def run_persona(state: AuditState, runtime: Runtime[AuditRuntimeContext]) -> dict[str, Any]:
        llm = runtime.context.llm.with_structured_output(PersonaReviewSchema)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PERSONA_PROMPTS[persona_key]),
                (
                    "human",
                    """
Audit the following tabular-model explanation from the perspective of the persona.

User query:
{user_query}

Instance:
{instance}

Explanation packet:
{explanation_packet}

Return a structured critique. Keep strengths and concerns specific to the evidence you see.
                    """.strip(),
                ),
            ]
        )
        chain = prompt | llm
        review = chain.invoke(
            {
                "user_query": state["user_query"],
                "instance": json.dumps(state["instance"], indent=2, sort_keys=True),
                "explanation_packet": json.dumps(state["explanation_packet"], indent=2, sort_keys=True),
            }
        )
        normalized = _normalize_result(review)
        normalized["persona"] = normalized.get("persona") or persona_name
        return {"persona_reviews": [normalized]}

    return run_persona


def _technical_assessment(
    state: AuditState,
    runtime: Runtime[AuditRuntimeContext],
) -> dict[str, Any]:
    explanation = state["explanation_packet"]
    technical = runtime.context.technical_assessor.assess(
        model=runtime.context.model,
        explainer=runtime.context.explainer,
        instance=state["instance"],
        explanation=_dict_to_explanation(explanation),
        reference_row=state.get("reference_row"),
        focus_label=state.get("focus_label"),
    )
    return {"technical_assessment": technical.to_dict()}


def _synthesize_report(state: AuditState, runtime: Runtime[AuditRuntimeContext]) -> dict[str, Any]:
    llm = runtime.context.llm.with_structured_output(FinalReportSchema)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are the audit coordinator synthesizing a final report about one tabular explanation method output.
Balance the stakeholder critiques with the technical metrics.

Verdict rules:
- pass: explanation is broadly usable and low-risk for this case
- review: explanation is partially useful but needs caution or revisions
- fail: explanation is actively misleading, unstable, or unfit for user-facing use
                """.strip(),
            ),
            (
                "human",
                """
Prepare the final audit report.

User query:
{user_query}

Instance:
{instance}

Explanation packet:
{explanation_packet}

Persona reviews:
{persona_reviews}

Technical assessment:
{technical_assessment}

Return a structured final report with a clear verdict and actionable recommendations.
                """.strip(),
            ),
        ]
    )
    chain = prompt | llm
    report = chain.invoke(
        {
            "user_query": state["user_query"],
            "instance": json.dumps(state["instance"], indent=2, sort_keys=True),
            "explanation_packet": json.dumps(state["explanation_packet"], indent=2, sort_keys=True),
            "persona_reviews": json.dumps(state.get("persona_reviews", []), indent=2, sort_keys=True),
            "technical_assessment": json.dumps(state.get("technical_assessment", {}), indent=2, sort_keys=True),
        }
    )
    normalized = _normalize_result(report)
    final_report = AuditReport(
        verdict=normalized["verdict"],
        executive_summary=normalized["executive_summary"],
        user_risks=normalized.get("user_risks", []),
        recommendations=normalized.get("recommendations", []),
        persona_summaries=[
            PersonaReview(**review) for review in state.get("persona_reviews", [])
        ],
        technical_assessment=_dict_to_technical_assessment(state.get("technical_assessment")),
    )
    return {"final_report": final_report.to_dict()}


def _normalize_result(result: Any) -> dict[str, Any]:
    if is_dataclass(result):
        return asdict(result)
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if isinstance(result, dict):
        return result
    raise TypeError(f"Unsupported structured output type: {type(result)!r}")


def _dict_to_explanation(payload: dict[str, Any]):
    from .types import LocalExplanation, PredictionResult

    prediction = PredictionResult(**payload["prediction"])
    return LocalExplanation(
        method_name=payload["method_name"],
        prediction=prediction,
        feature_attributions=payload["feature_attributions"],
        baseline_value=payload.get("baseline_value"),
        explanation_text=payload.get("explanation_text", ""),
        metadata=payload.get("metadata", {}),
    )


def _dict_to_technical_assessment(payload: dict[str, Any] | None):
    if not payload:
        return None
    from .types import TechnicalAssessment

    return TechnicalAssessment(**payload)
