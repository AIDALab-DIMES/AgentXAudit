from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Scalar = str | int | float | bool | None
TabularRow = dict[str, Scalar]


@dataclass
class PredictionResult:
    label: str
    score: float
    class_probabilities: dict[str, float] | None = None
    raw_output: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LocalExplanation:
    method_name: str
    prediction: PredictionResult
    feature_attributions: dict[str, float]
    baseline_value: float | None = None
    explanation_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PerturbationQuestion:
    question_id: str
    sample_id: str
    prompt: str
    perturbed_instance: TabularRow
    feature_name: str
    feature_change: str
    expected_label: str
    expected_score: float
    expected_score_direction: Literal["increase", "decrease", "no_change"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SampleAuditRecord:
    sample_id: str
    instance: TabularRow
    prediction: PredictionResult
    explanation: LocalExplanation
    perturbation_questions: list[PerturbationQuestion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PersonaQuestionAnswer:
    question_id: str
    predicted_label: str
    predicted_score_direction: Literal["increase", "decrease", "no_change"]
    reasoning: str = ""
    is_label_correct: bool = False
    is_direction_correct: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PersonaEvaluationInteraction:
    persona: str
    dimension: Literal["trust", "usability", "understanding"]
    coordinator_prompt: str
    persona_response: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PersonaReview:
    persona: str
    trust_score: int
    trust_rationale: str
    usability_score: int
    usability_rationale: str
    understanding_score: int
    understanding_summary: str
    strengths: list[str]
    concerns: list[str]
    question_answers: list[PersonaQuestionAnswer] = field(default_factory=list)
    scoring_breakdown: dict[str, Any] = field(default_factory=dict)
    interaction_history: list[PersonaEvaluationInteraction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TechnicalAssessment:
    reconstruction_fidelity: float | None
    directional_faithfulness: float | None
    stability: float | None
    notes: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AuditReport:
    verdict: Literal["pass", "review", "fail"]
    executive_summary: str
    aggregate_scores: dict[str, float]
    sampled_cases: list[SampleAuditRecord] = field(default_factory=list)
    persona_summaries: list[PersonaReview] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
