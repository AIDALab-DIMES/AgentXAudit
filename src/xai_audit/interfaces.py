from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from .types import LocalExplanation, TabularRow


@runtime_checkable
class TabularBlackBoxModel(Protocol):
    name: str
    feature_names: Sequence[str]

    def predict(self, rows: Sequence[TabularRow]) -> Sequence[Any]:
        """Return predicted labels or scalar outputs for each input row."""

    def predict_proba(self, rows: Sequence[TabularRow]) -> Sequence[Any]:
        """Optional probability-like outputs for each input row."""


@runtime_checkable
class FixedTabularExplainer(Protocol):
    name: str

    def explain_instance(
        self,
        model: TabularBlackBoxModel,
        instance: TabularRow,
        *,
        reference_row: TabularRow | None = None,
        focus_label: str | None = None,
    ) -> LocalExplanation:
        """Produce a local explanation for one tabular instance."""
