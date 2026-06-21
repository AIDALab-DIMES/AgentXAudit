from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.datasets import load_breast_cancer
from ucimlrepo import fetch_ucirepo


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    domain: str
    source_name: str
    source_url: str
    description: str
    positive_class: str
    categorical_features: tuple[str, ...] = ()
    loader_kind: str = "uci"
    loader_id: int | None = None
    notes: tuple[str, ...] = ()


@dataclass
class LoadedTabularDataset:
    spec: DatasetSpec
    features: pd.DataFrame
    target: pd.Series
    categorical_features: list[str]
    numeric_features: list[str]
    metadata: dict[str, object] = field(default_factory=dict)


DATASET_SPECS: dict[str, DatasetSpec] = {
    "german_credit": DatasetSpec(
        name="german_credit",
        domain="finance",
        source_name="UCI Machine Learning Repository",
        source_url="https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
        description="Binary credit-risk classification with 1000 applicants and 20 features.",
        positive_class="bad_credit_risk",
        loader_kind="uci",
        loader_id=144,
    ),
    "credit_card_default": DatasetSpec(
        name="credit_card_default",
        domain="finance",
        source_name="UCI Machine Learning Repository",
        source_url="https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
        description="Taiwan credit-card default prediction with 30000 clients and 23 tabular features.",
        positive_class="default_next_month",
        loader_kind="uci",
        loader_id=350,
        categorical_features=("X2", "X3", "X4", "X6", "X7", "X8", "X9", "X10", "X11"),
    ),
    "heart_disease": DatasetSpec(
        name="heart_disease",
        domain="healthcare",
        source_name="UCI Machine Learning Repository",
        source_url="https://archive.ics.uci.edu/dataset/45/heart+disease",
        description="Cleveland heart-disease diagnosis benchmark with 303 records and 13 core predictive features.",
        positive_class="heart_disease_present",
        loader_kind="uci",
        loader_id=45,
        categorical_features=("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"),
        notes=("The target is binarized from num > 0.",),
    ),
    "breast_cancer": DatasetSpec(
        name="breast_cancer",
        domain="healthcare",
        source_name="scikit-learn",
        source_url="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html",
        description="Wisconsin Diagnostic Breast Cancer binary classification dataset with 569 rows and 30 numeric features.",
        positive_class="malignant_breast_cancer",
        loader_kind="sklearn_builtin",
        notes=("The scikit-learn target is remapped so malignant becomes class 1.",),
    ),
}


def available_dataset_names() -> list[str]:
    return sorted(DATASET_SPECS)


def load_dataset(name: str) -> LoadedTabularDataset:
    if name not in DATASET_SPECS:
        raise KeyError(f"Unknown dataset {name!r}. Available: {', '.join(available_dataset_names())}")

    spec = DATASET_SPECS[name]
    if spec.loader_kind == "uci":
        return _load_uci_dataset(spec)
    if spec.loader_kind == "sklearn_builtin":
        return _load_breast_cancer_dataset(spec)
    raise ValueError(f"Unsupported loader kind: {spec.loader_kind}")


def _load_uci_dataset(spec: DatasetSpec) -> LoadedTabularDataset:
    if spec.loader_id is None:
        raise ValueError(f"Dataset {spec.name} is missing a UCI loader id.")

    fetched = fetch_ucirepo(id=spec.loader_id)
    features = fetched.data.features.copy()
    target_frame = fetched.data.targets.copy()
    target = target_frame.iloc[:, 0].copy()

    if spec.name == "german_credit":
        target = (
            target.astype(str)
            .str.strip()
            .map({"1": 0, "2": 1, "good": 0, "bad": 1})
            .astype("int64")
        )
    elif spec.name == "credit_card_default":
        target = pd.to_numeric(target, errors="raise").astype("int64")
        features = features.drop(columns=["ID"], errors="ignore")
    elif spec.name == "heart_disease":
        features = features.replace("?", pd.NA)
        features = features.apply(pd.to_numeric, errors="coerce")
        target = pd.to_numeric(target, errors="coerce").fillna(0).astype("int64")
        target = (target > 0).astype("int64")
    else:
        target = pd.to_numeric(target, errors="raise").astype("int64")

    inferred_categorical = [
        column
        for column in features.columns
        if str(features[column].dtype) in {"object", "string", "category"}
    ]
    categorical_features = sorted(set(column for column in spec.categorical_features if column in features.columns) | set(inferred_categorical))
    numeric_features = [column for column in features.columns if column not in categorical_features]

    metadata = {
        "domain": spec.domain,
        "source_name": spec.source_name,
        "source_url": spec.source_url,
        "description": spec.description,
        "positive_class": spec.positive_class,
        "notes": list(spec.notes),
        "n_rows": int(len(features)),
        "n_features": int(features.shape[1]),
    }
    return LoadedTabularDataset(
        spec=spec,
        features=features,
        target=pd.Series(target, name="target"),
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        metadata=metadata,
    )


def _load_breast_cancer_dataset(spec: DatasetSpec) -> LoadedTabularDataset:
    sklearn_bunch = load_breast_cancer(as_frame=True)
    features = sklearn_bunch.data.copy()
    original_target = sklearn_bunch.target.copy()
    target = (original_target == 0).astype("int64")

    metadata = {
        "domain": spec.domain,
        "source_name": spec.source_name,
        "source_url": spec.source_url,
        "description": spec.description,
        "positive_class": spec.positive_class,
        "target_names_original": list(sklearn_bunch.target_names),
        "notes": list(spec.notes),
        "n_rows": int(len(features)),
        "n_features": int(features.shape[1]),
    }
    return LoadedTabularDataset(
        spec=spec,
        features=features,
        target=pd.Series(target, name="target"),
        categorical_features=[],
        numeric_features=list(features.columns),
        metadata=metadata,
    )

