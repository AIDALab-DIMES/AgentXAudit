from .csv_export import build_csv_artifact
from .interfaces import FixedTabularExplainer, TabularBlackBoxModel
from .persona_processing import aggregate_scores, compose_persona_reviews
from .questionnaire import ValidationSamplingConfig, build_sample_audit_records
from .simple_xai import FiniteDifferenceTabularExplainer, ToyCreditRiskModel, build_demo_validation_set
from .technical import TechnicalAssessor, TechnicalAssessmentConfig
from .types import (
    AuditCsvArtifact,
    AuditCsvRow,
    LocalExplanation,
    PersonaEvaluationInteraction,
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

try:
    from .benchmarks import (
        DATASET_SPECS,
        DatasetSpec,
        LoadedTabularDataset,
        available_dataset_names,
        load_dataset,
        train_benchmark_dataset,
        train_many_benchmark_datasets,
    )
except ImportError:  # Optional until tabular benchmark dependencies are installed.
    DATASET_SPECS = None
    DatasetSpec = None
    LoadedTabularDataset = None
    available_dataset_names = None
    load_dataset = None
    train_benchmark_dataset = None
    train_many_benchmark_datasets = None

__all__ = [
    "AuditCoordinator",
    "AuditRuntimeContext",
    "AuditCsvArtifact",
    "AuditCsvRow",
    "DATASET_SPECS",
    "DatasetSpec",
    "FixedTabularExplainer",
    "FiniteDifferenceTabularExplainer",
    "LocalExplanation",
    "LoadedTabularDataset",
    "PersonaEvaluationInteraction",
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
    "aggregate_scores",
    "available_dataset_names",
    "build_csv_artifact",
    "build_demo_validation_set",
    "build_audit_graph",
    "build_sample_audit_records",
    "compose_persona_reviews",
    "load_dataset",
    "train_benchmark_dataset",
    "train_many_benchmark_datasets",
]
