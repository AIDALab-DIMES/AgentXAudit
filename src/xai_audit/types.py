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
    ground_truth_label: str | None = None
    prediction_is_correct: bool | None = None
    perturbation_questions: list[PerturbationQuestion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StakeholderProfile:
    stakeholder_id: str
    display_name: str
    role: str
    expertise_summary: str
    decision_context: str
    explanation_needs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PersonaTrustAnswer:
    question_id: str
    perceived_correct: bool
    actual_correct: bool = False
    is_alignment_correct: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PersonaUsabilityRating:
    question_id: str
    usability_score: int

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
    stakeholder_role: str
    dimension: Literal["trust", "usability", "understandability"]
    coordinator_prompt: str
    persona_response: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PersonaReview:
    persona: str
    stakeholder_role: str
    trust_score: float
    trust_rationale: str
    usability_score: float
    usability_rationale: str
    understandability_score: float
    understandability_summary: str
    strengths: list[str]
    concerns: list[str]
    trust_answers: list[PersonaTrustAnswer] = field(default_factory=list)
    usability_ratings: list[PersonaUsabilityRating] = field(default_factory=list)
    question_answers: list[PersonaQuestionAnswer] = field(default_factory=list)
    scoring_breakdown: dict[str, Any] = field(default_factory=dict)
    interaction_history: list[PersonaEvaluationInteraction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AuditCsvRow:
    row_type: Literal["stage", "interaction", "overall"]
    persona: str
    stakeholder_role: str
    stage: str
    aspect: str
    score: float | None = None
    summary: str = ""
    sample_id: str = ""
    question_id: str = ""
    perceived_correct: str = ""
    actual_correct: str = ""
    is_alignment_correct: str = ""
    usability_rating: float | None = None
    predicted_label: str = ""
    predicted_score_direction: str = ""
    is_label_correct: str = ""
    is_direction_correct: str = ""
    accuracy: float | None = None
    strengths: str = ""
    concerns: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AuditCsvArtifact:
    csv_content: str
    csv_rows: list[AuditCsvRow]
    aggregate_scores: dict[str, float]
    persona_summaries: list[PersonaReview] = field(default_factory=list)
    sampled_cases: list[SampleAuditRecord] = field(default_factory=list)
    csv_path: str | None = None

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
