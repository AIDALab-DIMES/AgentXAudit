from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from langchain.chat_models import init_chat_model

from xai_audit.artifact_adapters import ArtifactBackedBlackBoxModel
from xai_audit.benchmarks import available_dataset_names
from xai_audit.benchmarks.inference import load_trained_model_bundle
from xai_audit.explainers import ExplainerContext, available_explainers, build_explainer
from xai_audit.graph import AuditCoordinator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the audit framework on SHAP/LIME explanations dynamically produced for trained model artifacts."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=available_dataset_names(),
        help="Dataset artifact to audit.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help=f"Explanation methods to audit. Use 'all' or any of: {', '.join(available_explainers())}",
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts/benchmark_models",
        help="Directory containing trained benchmark model artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/audit_outputs",
        help="Directory where audit CSV outputs will be saved.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=5,
        help="Number of validation samples the audit coordinator should select.",
    )
    parser.add_argument(
        "--background-size",
        type=int,
        default=100,
        help="Number of background rows used by explanation methods.",
    )
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=10,
        help="Number of top local attributions to retain from each explanation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling reproducibility.",
    )
    parser.add_argument(
        "--audit-llm-model",
        default=None,
        help="Optional LangChain model string. Falls back to AUDIT_LLM_MODEL or openai:gpt-4.1-mini.",
    )
    parser.add_argument(
        "--focus-label",
        default=None,
        help="Positive class label used for score-focused auditing. Defaults to the dataset positive class from metadata.",
    )
    parser.add_argument(
        "--scenario-description",
        default=None,
        help="Optional deployment scenario used by the audit agent to plan the stakeholder personas.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    llm_name = args.audit_llm_model or "openai:gpt-4.1-mini"
    llm = init_chat_model(llm_name)

    bundle = load_trained_model_bundle(dataset_name=args.dataset, artifact_root=args.artifact_root)
    focus_label = args.focus_label or str(bundle.metadata.get("positive_class", "positive"))
    scenario_description = args.scenario_description or _default_scenario_description(
        dataset_name=args.dataset,
        metadata=bundle.metadata,
    )
    model = ArtifactBackedBlackBoxModel(
        bundle=bundle,
        positive_label=focus_label,
        negative_label="not_" + focus_label,
        name=f"{args.dataset}_artifact_model",
    )

    dataset_csv_path = Path(bundle.metadata["artifact_files"]["dataset_csv"])
    validation_csv_path = Path(bundle.metadata["artifact_files"]["test_split_csv"])
    validation_frame = pd.read_csv(validation_csv_path)
    dataset_frame = pd.read_csv(dataset_csv_path)
    feature_names = bundle.feature_names
    validation_columns = feature_names + [column for column in ("target",) if column in validation_frame.columns]
    validation_set = validation_frame[validation_columns].to_dict(orient="records")
    background_data = dataset_frame[feature_names].sample(
        n=min(args.background_size, len(dataset_frame)),
        random_state=args.random_state,
    ).reset_index(drop=True)

    method_names = available_explainers() if args.methods == ["all"] else args.methods
    coordinator = AuditCoordinator(llm=llm, sample_count=args.sample_count)

    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest: list[dict[str, str]] = []

    for method_name in method_names:
        context = ExplainerContext(
            bundle=bundle,
            background_data=background_data,
            feature_names=feature_names,
            class_names=bundle.class_names,
            top_k_features=args.top_k_features,
            random_state=args.random_state,
        )
        explainer = build_explainer(method_name, context=context)
        csv_path = output_dir / f"audit_{method_name}.csv"
        result = coordinator.audit(
            user_query=(
                f"Audit the {method_name.upper()} explanations produced for the trained {args.dataset} tabular model. "
                "Assess trust calibration, usability for downstream decisions, and understandability for the involved stakeholders."
            ),
            scenario_description=scenario_description,
            validation_set=validation_set,
            model=model,
            explainer=explainer,
            focus_label=focus_label,
            sample_count=args.sample_count,
            csv_output_path=str(csv_path),
        )
        run_manifest.append(
            {
                "method_name": method_name,
                "csv_path": str(Path(result["csv_artifact"]["csv_path"]).resolve()),
            }
        )

    manifest_path = output_dir / "audit_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest_path": str(manifest_path.resolve()), "methods": method_names}, indent=2))
    return 0


def _default_scenario_description(*, dataset_name: str, metadata: dict[str, object]) -> str:
    domain = str(metadata.get("domain", "")).strip().lower()
    positive_class = str(metadata.get("positive_class", "positive"))
    if domain == "healthcare":
        return (
            f"A healthcare provider uses the {dataset_name} model to flag patients at risk of {positive_class}. "
            "The explanations are shown to a clinician who decides follow-up tests or treatment changes and to the patient, "
            "who uses the explanation to understand risk factors and recommended behavioral changes."
        )
    if domain == "finance":
        return (
            f"A financial institution uses the {dataset_name} model to support decisions related to {positive_class}. "
            "The explanations are shown to an internal decision maker who decides follow-up review actions and to the affected customer, "
            "who uses the explanation to understand the decision and possible next steps."
        )
    return (
        f"A tabular classification model for {dataset_name} predicts {positive_class}. "
        "Its explanations are provided both to a domain expert who acts on the prediction and to an affected end user."
    )


if __name__ == "__main__":
    raise SystemExit(main())
