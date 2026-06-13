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
class PersonaReview:
    persona: str
    summary: str
    strengths: list[str]
    concerns: list[str]
    follow_up_questions: list[str]
    trust_level: Literal["low", "medium", "high"]

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
    user_risks: list[str]
    recommendations: list[str]
    persona_summaries: list[PersonaReview] = field(default_factory=list)
    technical_assessment: TechnicalAssessment | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
