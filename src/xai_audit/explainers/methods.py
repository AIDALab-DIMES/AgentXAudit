from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from xai_audit.model_utils import predict_single
from xai_audit.types import LocalExplanation, TabularRow

from .base import BaseAuditExplainer, register_explainer, top_attributions


@register_explainer("shap")
@dataclass
class ShapAuditExplainer(BaseAuditExplainer):
    max_evals: int = 250

    def __init__(self, *, context, max_evals: int = 250) -> None:
        super().__init__(context=context)
        self.max_evals = max_evals

    def explain_instance(
        self,
        model,
        instance: TabularRow,
        *,
        reference_row: TabularRow | None = None,
        focus_label: str | None = None,
    ) -> LocalExplanation:
        import shap

        instance_frame = pd.DataFrame([instance], columns=self.context.feature_names)
        prediction = predict_single(model, instance, focus_label=focus_label)
        explainer = shap.Explainer(
            self.context.predict_positive_probability,
            self.context.background_data[self.context.feature_names],
            algorithm="permutation",
            feature_names=self.context.feature_names,
            seed=self.context.random_state,
        )
        explanation = explainer(instance_frame, max_evals=self.max_evals)
        values = explanation.values[0]
        base_value = explanation.base_values[0] if getattr(explanation.base_values, "__len__", None) else explanation.base_values
        attributions = top_attributions(self.context.feature_names, values, self.context.top_k_features)
        return LocalExplanation(
            method_name=self.name,
            prediction=prediction,
            feature_attributions={name: round(value, 6) for name, value in attributions.items()},
            baseline_value=float(base_value),
            explanation_text=_render_text(attributions, prediction.label),
            metadata={"algorithm": "shap_permutation", "all_feature_names": self.context.feature_names},
        )


@register_explainer("lime")
@dataclass
class LimeAuditExplainer(BaseAuditExplainer):
    num_samples: int = 2000

    def __init__(self, *, context, num_samples: int = 2000) -> None:
        super().__init__(context=context)
        self.num_samples = num_samples

    def explain_instance(
        self,
        model,
        instance: TabularRow,
        *,
        reference_row: TabularRow | None = None,
        focus_label: str | None = None,
    ) -> LocalExplanation:
        from lime.lime_tabular import LimeTabularExplainer

        instance_frame = pd.DataFrame([instance], columns=self.context.feature_names)
        prediction = predict_single(model, instance, focus_label=focus_label)
        encoded_instance = self.context.encode_for_lime(instance_frame)
        explainer = LimeTabularExplainer(
            training_data=self.context.lime_training_data(),
            mode="classification",
            feature_names=self.context.feature_names,
            categorical_features=self.context.categorical_feature_indices,
            categorical_names=self.context.categorical_names,
            class_names=self.context.class_names,
            discretize_continuous=False,
            random_state=self.context.random_state,
        )
        explanation = explainer.explain_instance(
            data_row=encoded_instance.iloc[0].to_numpy(),
            predict_fn=lambda values: self.context.predict_class_probabilities(self.context.decode_lime_array(values)),
            labels=(1,),
            num_features=min(self.context.top_k_features, len(self.context.feature_names)),
            num_samples=self.num_samples,
        )
        local_exp = sorted(explanation.local_exp[1], key=lambda item: abs(item[1]), reverse=True)
        attributions = {
            self.context.feature_names[int(feature_index)]: float(weight)
            for feature_index, weight in local_exp[: self.context.top_k_features]
        }
        intercept = explanation.intercept
        if isinstance(intercept, dict):
            base_value = float(intercept.get(1))
        elif isinstance(intercept, list):
            base_value = float(intercept[1] if len(intercept) > 1 else intercept[0])
        else:
            base_value = float(intercept)
        return LocalExplanation(
            method_name=self.name,
            prediction=prediction,
            feature_attributions={name: round(value, 6) for name, value in attributions.items()},
            baseline_value=base_value,
            explanation_text=_render_text(attributions, prediction.label),
            metadata={"algorithm": "lime_tabular", "all_feature_names": self.context.feature_names},
        )


def _render_text(attributions: dict[str, float], label: str) -> str:
    if not attributions:
        return f"The explainer did not return usable attributions."
    clauses = []
    for feature, value in list(attributions.items())[:3]:
        direction = "positively affects" if value > 0 else "negatively affects"
        clauses.append(f"{feature} {direction} the observed prediction with an importance of {abs(value):.3f}")
    return f"{'; '.join(clauses)}." # Predicted label: {label}. 
