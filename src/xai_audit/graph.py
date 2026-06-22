from __future__ import annotations

import operator
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Annotated, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from .csv_export import build_csv_artifact
from .interfaces import FixedTabularExplainer, TabularBlackBoxModel
from .persona_processing import aggregate_scores, compose_persona_reviews
from .questionnaire import (
    ValidationSamplingConfig,
    build_persona_stage_packet,
    build_sample_audit_records,
)
from .types import StakeholderProfile, TabularRow


AUDIT_DIMENSIONS: tuple[Literal["trust", "usability", "understandability"], ...] = (
    "trust",
    "usability",
    "understandability",
)


class AuditState(TypedDict, total=False):
    user_query: str
    scenario_description: str | None
    validation_set: list[TabularRow]
    focus_label: str | None
    sample_count: int
    csv_output_path: str | None
    stakeholder_profiles: list[dict[str, Any]]
    workflow_steps: list[dict[str, str]]
    current_step_index: int
    sampled_cases: list[dict[str, Any]]
    interaction_records: Annotated[list[dict[str, Any]], operator.add]
    csv_artifact: dict[str, Any]


@dataclass
class AuditRuntimeContext:
    llm: BaseChatModel
    model: TabularBlackBoxModel
    explainer: FixedTabularExplainer


@dataclass
class StakeholderProfileSchema:
    stakeholder_id: str
    display_name: str
    role: str
    expertise_summary: str
    decision_context: str
    explanation_needs: list[str]
    narrative_description: str


@dataclass
class StakeholderPlanningSchema:
    stakeholders: list[StakeholderProfileSchema]


@dataclass
class TrustSampleAssessmentSchema:
    persona: str
    question_id: str
    perceived_correct: bool


@dataclass
class UsabilitySampleAssessmentSchema:
    persona: str
    question_id: str
    usability_score: int


@dataclass
class UnderstandabilityQuestionAnswerSchema:
    persona: str
    question_id: str
    predicted_label: str
    predicted_score_direction: Literal["increase", "decrease", "no_change"]


DIMENSION_INSTRUCTIONS = {
    "trust": """
This interaction evaluates trust only.
Judge whether the model prediction appears correct for the presented sample solely relying on the provided explanation.
Return only whether you believe the prediction is correct or not.
""".strip(),
    "usability": """
This interaction evaluates usability only.
Judge whether the explanation supports downstream decisions or actions for this stakeholder on the presented sample.
Return a usability score from 1 to 5, where 1 means not usable and 5 means highly usable.
""".strip(),
    "understandability": """
This interaction evaluates understandability only.
Use the explanation to predict how the black-box model would behave under the provided perturbation.
Return the predicted label and whether the focal score would increase, decrease, or stay about the same.
""".strip(),
}


def build_audit_graph() -> Any:
    builder = StateGraph(AuditState, context_schema=AuditRuntimeContext)
    builder.add_node("initialize_audit", _initialize_audit)
    builder.add_node("plan_stakeholders", _plan_stakeholders)
    builder.add_node("run_next_stage", _run_next_stage)
    builder.add_node("build_csv_output", _build_csv_output)

    builder.add_edge(START, "initialize_audit")
    builder.add_edge("initialize_audit", "plan_stakeholders")
    builder.add_conditional_edges(
        "plan_stakeholders",
        _route_after_planning,
        {
            "run_next_stage": "run_next_stage",
            "build_csv_output": "build_csv_output",
        },
    )
    builder.add_conditional_edges(
        "run_next_stage",
        _route_after_stage,
        {
            "run_next_stage": "run_next_stage",
            "build_csv_output": "build_csv_output",
        },
    )
    builder.add_edge("build_csv_output", END)
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
        csv_output_path: str | None = None,
        scenario_description: str | None = None,
        stakeholder_profiles: list[StakeholderProfile] | None = None,
    ) -> dict[str, Any]:
        context = AuditRuntimeContext(
            llm=self._llm,
            model=model,
            explainer=explainer,
        )
        initial_state: AuditState = {
            "user_query": user_query,
            "scenario_description": scenario_description,
            "validation_set": validation_set,
            "focus_label": focus_label,
            "sample_count": sample_count or self._sample_count,
            "csv_output_path": csv_output_path,
            "stakeholder_profiles": [profile.to_dict() for profile in stakeholder_profiles or []],
            "workflow_steps": [],
            "current_step_index": 0,
            "interaction_records": [],
        }
        return self._graph.invoke(initial_state, context=context)


def _initialize_audit(state: AuditState, runtime: Runtime[AuditRuntimeContext]) -> dict[str, Any]:
    records = build_sample_audit_records(
        model=runtime.context.model,
        explainer=runtime.context.explainer,
        validation_set=state["validation_set"],
        config=ValidationSamplingConfig(sample_count=state["sample_count"]),
        focus_label=state.get("focus_label"),
    )
    return {"sampled_cases": [record.to_dict() for record in records]}


def _plan_stakeholders(state: AuditState, runtime: Runtime[AuditRuntimeContext]) -> dict[str, Any]:
    if state.get("stakeholder_profiles"):
        stakeholders = _coerce_stakeholder_profiles(state["stakeholder_profiles"])
    elif state.get("scenario_description"):
        stakeholders = _infer_stakeholders_with_llm(state, runtime)
    else:
        stakeholders = _default_stakeholder_profiles()

    workflow_steps = [
        {
            "stakeholder_id": profile.stakeholder_id,
            "dimension": dimension,
        }
        for profile in stakeholders
        for dimension in AUDIT_DIMENSIONS
    ]
    return {
        "stakeholder_profiles": [profile.to_dict() for profile in stakeholders],
        "workflow_steps": workflow_steps,
        "current_step_index": 0,
    }


def _infer_stakeholders_with_llm(
    state: AuditState,
    runtime: Runtime[AuditRuntimeContext],
) -> list[StakeholderProfile]:
    planner_llm = _bounded_structured_llm(runtime.context.llm, StakeholderPlanningSchema, max_tokens=600)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are the audit agent orchestrating a human-centered XAI assessment.
From the deployment scenario, identify the stakeholder categories that directly receive explanations or act on them.
Distinguish stakeholders with different expertise and decision responsibilities.
Additionally, for each stakeholder, produce a brief narrative description of the user to impersonate, summarizing all its characteristics and its needs. Be concise in the description.

Keep stakeholder_id short and snake_case.
""".strip(),
            ),
            (
                "human",
                """
User query:
{user_query}

Deployment scenario:
{scenario_description}

Return only the stakeholders that should participate in the audit of trust, usability, and understandability.
""".strip(),
            ),
        ]
    )
    result = _normalize_result(
        (prompt | planner_llm).invoke(
            {
                "user_query": state["user_query"],
                "scenario_description": state["scenario_description"],
            }
        )
    )
    stakeholders = _coerce_stakeholder_profiles(result.get("stakeholders", []))
    return stakeholders or _default_stakeholder_profiles()


def _run_next_stage(state: AuditState, runtime: Runtime[AuditRuntimeContext]) -> dict[str, Any]:
    workflow_steps = state.get("workflow_steps", [])
    current_step_index = state.get("current_step_index", 0)
    if current_step_index >= len(workflow_steps):
        return {"current_step_index": current_step_index}

    step = workflow_steps[current_step_index]
    stakeholders = _coerce_stakeholder_profiles(state.get("stakeholder_profiles", []))
    stakeholder = _find_stakeholder_profile(stakeholders, step["stakeholder_id"])
    dimension = step["dimension"]

    if dimension == "trust":
        payload = _run_trust_stage(stakeholder, state, runtime)
    elif dimension == "usability":
        payload = _run_usability_stage(stakeholder, state, runtime)
    else:
        payload = _run_understandability_stage(stakeholder, state, runtime)

    payload["current_step_index"] = current_step_index + 1
    return payload


def _run_trust_stage(
    stakeholder: StakeholderProfile,
    state: AuditState,
    runtime: Runtime[AuditRuntimeContext],
) -> dict[str, Any]:
    llm = _bounded_structured_llm(runtime.context.llm, TrustSampleAssessmentSchema, max_tokens=80)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _build_stakeholder_system_prompt(stakeholder, "trust", state.get("scenario_description"))),
            (
                "human",
                """
Coordinator request:
{coordinator_prompt}
                """.strip(),
            ),
        ]
    )

    interaction_records: list[dict[str, Any]] = []
    for record in _dicts_to_sample_records(state["sampled_cases"]):
        question_id = f"{record.sample_id}_trust"
        sample_packet = build_persona_stage_packet([record], dimension="trust")
        print(f'[SAMPLE PACKET]: {sample_packet}\n----------------------------')
        coordinator_prompt = _build_coordinator_prompt(
            stakeholder=stakeholder,
            dimension="trust",
            user_query=state["user_query"],
            #scenario_description=state.get("scenario_description"),
            sampled_cases_text=sample_packet,
            question_text=(
                f"Question id: {question_id}\n"
                "Based only on the sample, the prediction, and the explanation, do you believe the model prediction is correct for this case?"
            ),
        )
        print(f'[COORDINATOR PROMPT]: {coordinator_prompt}\n-----------------------------')
        fprompt = prompt.format_messages(coordinator_prompt=coordinator_prompt)##invoke({"coordinator_prompt": coordinator_prompt})
        print(f'[FINAL PROMPT]:\n {fprompt}\n---------------------')
        response = _normalize_result(llm.invoke(fprompt))
        response["persona"] = stakeholder.display_name
        response["question_id"] = response.get("question_id") or question_id
        print(f'[PERSONA RESPONSE]: {response}\n-------------------------')
        
        interaction_records.append(
            {
                "persona": response["persona"],
                "stakeholder_role": stakeholder.role,
                "dimension": "trust",
                "coordinator_prompt": coordinator_prompt,
                "persona_response": response,
            }
        )
    return {"interaction_records": interaction_records}


def _run_usability_stage(
    stakeholder: StakeholderProfile,
    state: AuditState,
    runtime: Runtime[AuditRuntimeContext],
) -> dict[str, Any]:
    llm = _bounded_structured_llm(runtime.context.llm, UsabilitySampleAssessmentSchema, max_tokens=80)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _build_stakeholder_system_prompt(stakeholder, "usability", state.get("scenario_description"))),
            (
                "human",
                """
                
Coordinator request:
{coordinator_prompt}
                """.strip(),
            ),
        ]
    )

    interaction_records: list[dict[str, Any]] = []
    for record in _dicts_to_sample_records(state["sampled_cases"]):
        question_id = f"{record.sample_id}_usability"
        sample_packet = build_persona_stage_packet([record], dimension="usability")
        coordinator_prompt = _build_coordinator_prompt(
            stakeholder=stakeholder,
            dimension="usability",
            user_query=state["user_query"],
            #scenario_description=state.get("scenario_description"),
            sampled_cases_text=sample_packet,
            question_text=(
                f"Question id: {question_id}\n"
                "How usable is this explanation for supporting your next decision or action on this case?"
            ),
        )
        response = _normalize_result((prompt | llm).invoke({"coordinator_prompt": coordinator_prompt}))
        response["persona"] = stakeholder.display_name
        response["question_id"] = response.get("question_id") or question_id
        interaction_records.append(
            {
                "persona": response["persona"],
                "stakeholder_role": stakeholder.role,
                "dimension": "usability",
                "coordinator_prompt": coordinator_prompt,
                "persona_response": response,
            }
        )
    return {"interaction_records": interaction_records}


def _run_understandability_stage(
    stakeholder: StakeholderProfile,
    state: AuditState,
    runtime: Runtime[AuditRuntimeContext],
) -> dict[str, Any]:
    llm = _bounded_structured_llm(runtime.context.llm, UnderstandabilityQuestionAnswerSchema, max_tokens=60)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _build_stakeholder_system_prompt(stakeholder, "understandability", state.get("scenario_description"))),
            (
                "human",
                """

Coordinator request:
{coordinator_prompt}
                """.strip(),
            ),
        ]
    )

    interaction_records: list[dict[str, Any]] = []
    for record in _dicts_to_sample_records(state["sampled_cases"]):
        if not record.perturbation_questions:
            continue
        question_packet = build_persona_stage_packet([record], dimension="understandability")
        question_request = _build_coordinator_prompt(
            stakeholder=stakeholder,
            dimension="understandability",
            user_query=state["user_query"],
            #scenario_description=state.get("scenario_description"),
            sampled_cases_text=question_packet,
            question_text="Answer only the single perturbation question shown in the case packet.",
        )
        response = _normalize_result((prompt | llm).invoke({"coordinator_prompt": question_request}))
        response["persona"] = stakeholder.display_name
        interaction_records.append(
            {
                "persona": response["persona"],
                "stakeholder_role": stakeholder.role,
                "dimension": "understandability",
                "coordinator_prompt": question_request,
                "persona_response": response,
            }
        )
    return {"interaction_records": interaction_records}


def _build_csv_output(state: AuditState, runtime: Runtime[AuditRuntimeContext]) -> dict[str, Any]:
    persona_reviews = compose_persona_reviews(
        state.get("interaction_records", []),
        state.get("sampled_cases", []),
    )
    aggregate = aggregate_scores(persona_reviews)
    csv_artifact = build_csv_artifact(
        interaction_records=state.get("interaction_records", []),
        persona_reviews=persona_reviews,
        aggregate_scores=aggregate,
        sampled_cases=_dicts_to_sample_records(state.get("sampled_cases", [])),
        csv_path=state.get("csv_output_path"),
    )
    return {"csv_artifact": csv_artifact.to_dict()}


def _route_after_planning(state: AuditState) -> str:
    return "run_next_stage" if state.get("workflow_steps") else "build_csv_output"


def _route_after_stage(state: AuditState) -> str:
    current_step_index = state.get("current_step_index", 0)
    workflow_steps = state.get("workflow_steps", [])
    return "build_csv_output" if current_step_index >= len(workflow_steps) else "run_next_stage"


def _build_coordinator_prompt(
    *,
    stakeholder: StakeholderProfile,
    dimension: Literal["trust", "usability", "understandability"],
    user_query: str,
    #scenario_description: str | None,
    sampled_cases_text: str,
    question_text: str,
) -> str:
    return (
        #f"User query:\n{user_query}\n\n"
        #f"Deployment scenario:\n{scenario_description or 'Not provided.'}\n\n"
        #f"You are impersonating the following user:\n{stakeholder.narrative_description}\n\n"#{_format_stakeholder_block(stakeholder)}\n\n"
        #f"Evaluation dimension:\n{dimension}\n\n"
        #f"Stage instructions:\n{DIMENSION_INSTRUCTIONS[dimension]}\n\n"
        #"Ground your assessment only in the provided case.\n\n"
        f"Case packet:\n{sampled_cases_text}\n\n"
        f"{question_text}"
    )


def _build_stakeholder_system_prompt(
    stakeholder: StakeholderProfile,
    dimension: Literal["trust", "usability", "understandability"],
    scenario_description: str | None,
) -> str:
    explanation_needs = "; ".join(stakeholder.explanation_needs) or "No specific needs provided."
    return (
        f"You are simulating the following user. {stakeholder.narrative_description}\n"#{stakeholder.display_name}, acting as a {stakeholder.role}.\n"
        #f"Scenario context: {scenario_description or 'Not provided.'}\n"
        #f"Expertise: {stakeholder.expertise_summary}\n"
        #f"Decision context: {stakeholder.decision_context}\n"
        #f"Explanation needs: {explanation_needs}\n"
        "You are interacting with the audit coordinator in one stage of a multi-step evaluation."

        f"Stage instructions:\n{DIMENSION_INSTRUCTIONS[dimension]}\n\n"
        
        "Ground all judgments only in the provided case packet.\n"
        "Stay consistent with this stakeholder's expertise, responsibilities, and information needs."
    )


def _format_stakeholder_block(stakeholder: StakeholderProfile) -> str:
    needs = "; ".join(stakeholder.explanation_needs) or "No specific explanation needs provided."
    return (
        f"name={stakeholder.display_name}\n"
        f"role={stakeholder.role}\n"
        f"expertise={stakeholder.expertise_summary}\n"
        f"decision_context={stakeholder.decision_context}\n"
        f"explanation_needs={needs}"
    )


def _coerce_stakeholder_profiles(payload: list[dict[str, Any]] | list[StakeholderProfile]) -> list[StakeholderProfile]:
    output: list[StakeholderProfile] = []
    for index, item in enumerate(payload, start=1):
        if isinstance(item, StakeholderProfile):
            output.append(item)
            continue
        stakeholder_id = str(item.get("stakeholder_id") or f"stakeholder_{index}").strip().lower().replace(" ", "_")
        output.append(
            StakeholderProfile(
                stakeholder_id=stakeholder_id,
                display_name=str(item.get("display_name") or item.get("persona") or stakeholder_id.replace("_", " ").title()),
                role=str(item.get("role") or "stakeholder"),
                expertise_summary=str(item.get("expertise_summary") or "General familiarity with the use case."),
                decision_context=str(item.get("decision_context") or "Receives explanations and reacts to them."),
                explanation_needs=[str(entry) for entry in item.get("explanation_needs", [])],
                narrative_description = str(item.get("narrative_description"))
            )
        )
    return output


def _default_stakeholder_profiles() -> list[StakeholderProfile]:
    return [
        StakeholderProfile(
            stakeholder_id="internal_model_reviewer",
            display_name="Internal Model Reviewer",
            role="technical reviewer",
            expertise_summary="Can inspect predictive behavior and explanation consistency, but still needs clear evidence.",
            decision_context="Reviews whether the explanation is credible enough to support model deployment.",
            explanation_needs=[
                "evidence that explanations match model behavior",
                "signals of failure modes or instability",
            ],
        ),
        StakeholderProfile(
            stakeholder_id="domain_decision_maker",
            display_name="Domain Decision Maker",
            role="operational decision maker",
            expertise_summary="Understands the application domain and uses explanations to decide follow-up actions.",
            decision_context="Uses explanations to decide what action should follow the model prediction.",
            explanation_needs=[
                "clear rationale for the prediction",
                "guidance for what action to take next",
            ],
        ),
        StakeholderProfile(
            stakeholder_id="affected_end_user",
            display_name="Affected End User",
            role="affected individual",
            expertise_summary="Limited technical expertise and wants a clear, fair, and actionable explanation.",
            decision_context="Receives the explanation as a person affected by the model outcome.",
            explanation_needs=[
                "plain-language explanation",
                "understanding of why the decision happened",
                "clarity on possible next steps",
            ],
        ),
    ]


def _find_stakeholder_profile(
    stakeholders: list[StakeholderProfile],
    stakeholder_id: str,
) -> StakeholderProfile:
    for stakeholder in stakeholders:
        if stakeholder.stakeholder_id == stakeholder_id:
            return stakeholder
    raise KeyError(f"Unknown stakeholder_id {stakeholder_id!r}")


def _normalize_result(result: Any) -> dict[str, Any]:
    if is_dataclass(result):
        return asdict(result)
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if isinstance(result, dict):
        return result
    raise TypeError(f"Unsupported structured output type: {type(result)!r}")


def _bounded_structured_llm(llm: BaseChatModel, schema: type[Any], max_tokens: int):
    bound = llm
    for key in ("max_completion_tokens", "max_tokens"):
        try:
            bound = bound.bind(**{key: max_tokens})
            break
        except Exception:
            continue
    return bound.with_structured_output(schema)


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
                ground_truth_label=item.get("ground_truth_label"),
                prediction_is_correct=item.get("prediction_is_correct"),
                perturbation_questions=questions,
            )
        )
    return records
