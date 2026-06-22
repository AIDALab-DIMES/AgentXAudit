from .csv_export import build_csv_artifact
from .artifact_adapters import ArtifactBackedBlackBoxModel
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
    PersonaTrustAnswer,
    PersonaUsabilityRating,
    PerturbationQuestion,
    PredictionResult,
    SampleAuditRecord,
    StakeholderProfile,
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
        TrainedModelBundle,
        available_dataset_names,
        load_dataset,
        load_trained_model_bundle,
        train_benchmark_dataset,
        train_many_benchmark_datasets,
    )
except ImportError:  # Optional until tabular benchmark dependencies are installed.
    DATASET_SPECS = None
    DatasetSpec = None
    LoadedTabularDataset = None
    TrainedModelBundle = None
    available_dataset_names = None
    load_dataset = None
    load_trained_model_bundle = None
    train_benchmark_dataset = None
    train_many_benchmark_datasets = None

try:
    from .explainers import (
        AnchorAuditExplainer,
        BaseAuditExplainer,
        ExplainerContext,
        LimeAuditExplainer,
        ShapAuditExplainer,
        available_explainers,
        build_explainer,
        register_explainer,
    )
except ImportError:  # Optional until explanation dependencies are installed.
    AnchorAuditExplainer = None
    BaseAuditExplainer = None
    ExplainerContext = None
    LimeAuditExplainer = None
    ShapAuditExplainer = None
    available_explainers = None
    build_explainer = None
    register_explainer = None

__all__ = [
    "AuditCoordinator",
    "AuditRuntimeContext",
    "AuditCsvArtifact",
    "AuditCsvRow",
    "AnchorAuditExplainer",
    "ArtifactBackedBlackBoxModel",
    "BaseAuditExplainer",
    "DATASET_SPECS",
    "DatasetSpec",
    "ExplainerContext",
    "FixedTabularExplainer",
    "FiniteDifferenceTabularExplainer",
    "LimeAuditExplainer",
    "LocalExplanation",
    "LoadedTabularDataset",
    "PersonaEvaluationInteraction",
    "PersonaQuestionAnswer",
    "PersonaReview",
    "PersonaTrustAnswer",
    "PersonaUsabilityRating",
    "PerturbationQuestion",
    "PredictionResult",
    "SampleAuditRecord",
    "ShapAuditExplainer",
    "TabularBlackBoxModel",
    "TechnicalAssessment",
    "TechnicalAssessmentConfig",
    "TechnicalAssessor",
    "TrainedModelBundle",
    "ToyCreditRiskModel",
    "StakeholderProfile",
    "ValidationSamplingConfig",
    "aggregate_scores",
    "available_dataset_names",
    "available_explainers",
    "build_csv_artifact",
    "build_demo_validation_set",
    "build_audit_graph",
    "build_explainer",
    "build_sample_audit_records",
    "compose_persona_reviews",
    "load_dataset",
    "load_trained_model_bundle",
    "register_explainer",
    "train_benchmark_dataset",
    "train_many_benchmark_datasets",
]
