from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from .benchmarks.inference import TrainedModelBundle
from .interfaces import TabularBlackBoxModel
from .types import TabularRow


@dataclass
class ArtifactBackedBlackBoxModel:
    bundle: TrainedModelBundle
    positive_label: str = "approve"
    negative_label: str = "reject"
    name: str = "artifact_backed_black_box"

    @property
    def feature_names(self) -> Sequence[str]:
        return tuple(self.bundle.feature_names)

    @property
    def classes_(self) -> tuple[str, str]:
        return (self.negative_label, self.positive_label)

    def predict(self, rows: Sequence[TabularRow]) -> list[str]:
        frame = pd.DataFrame(rows, columns=self.bundle.feature_names)
        labels = self.bundle.predict(frame)
        return [self.positive_label if int(label) == 1 else self.negative_label for label in labels]

    def predict_proba(self, rows: Sequence[TabularRow]) -> list[dict[str, float]]:
        frame = pd.DataFrame(rows, columns=self.bundle.feature_names)
        probabilities = self.bundle.predict_proba(frame)
        return [
            {
                self.negative_label: float(probability[0]),
                self.positive_label: float(probability[1]),
            }
            for probability in probabilities
        ]
