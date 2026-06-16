from .interfaces import FixedTabularExplainer, TabularBlackBoxModel
from .questionnaire import ValidationSamplingConfig, build_sample_audit_records
from .simple_xai import FiniteDifferenceTabularExplainer, ToyCreditRiskModel, build_demo_validation_set
from .technical import TechnicalAssessor, TechnicalAssessmentConfig
from .types import (
    AuditReport,
    LocalExplanation,
    PersonaQuestionAnswer,
    PersonaReview,
    PerturbationQuestion,
    PredictionResult,
    SampleAuditRecord,
    TechnicalAssessment,
)

try:
    from .graph import AuditCoordinator, AuditRuntimeContext, build_audit_graph
except ImportError:  # Optional until LangChain/LangGraph dependencies are installed.
    AuditCoordinator = None
    AuditRuntimeContext = None
    build_audit_graph = None

__all__ = [
    "AuditCoordinator",
    "AuditRuntimeContext",
    "AuditReport",
    "FixedTabularExplainer",
    "FiniteDifferenceTabularExplainer",
    "LocalExplanation",
    "PersonaQuestionAnswer",
    "PersonaReview",
    "PerturbationQuestion",
    "PredictionResult",
    "SampleAuditRecord",
    "TabularBlackBoxModel",
    "TechnicalAssessment",
    "TechnicalAssessmentConfig",
    "TechnicalAssessor",
    "ToyCreditRiskModel",
    "ValidationSamplingConfig",
    "build_demo_validation_set",
    "build_audit_graph",
    "build_sample_audit_records",
]
