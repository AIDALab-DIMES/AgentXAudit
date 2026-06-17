# XAI Audit Prototype

This project implements a prototype framework for auditing the explanations emitted by a fixed explainability method on tabular data.

The current workflow is validation-set driven and iterative:

1. The LangGraph coordinator receives:
   - a black-box tabular model
   - a fixed explanation algorithm
   - a validation set
2. The coordinator selects between 5 and 10 representative samples from the validation set.
3. For each selected sample, the framework extracts:
   - the black-box model output
   - the explanation method output
   - a perturbation question whose answer can be checked against the black-box model
4. Three persona agents review the same sampled evidence:
   - data scientist
   - business executive
   - end user
5. For each persona, the audit coordinator runs three separate interactions over the same validation samples:
   - trust evaluation only
   - usability evaluation only
   - understanding evaluation only
6. Understanding is measured by how well the persona predicts the black-box model output on perturbed samples.
7. The coordinator exports a CSV with staged persona outcomes and overall aspect scores.

The orchestration layer uses `LangGraph`, while LLM interactions use `LangChain`.

## Project Layout

- `src/xai_audit/interfaces.py`: black-box model and explainer protocols
- `src/xai_audit/model_utils.py`: helpers for model predictions and score extraction
- `src/xai_audit/questionnaire.py`: validation-set sampling and perturbation-question generation
- `src/xai_audit/persona_processing.py`: iterative persona-response grading and aggregation
- `src/xai_audit/csv_export.py`: deterministic CSV generation for stage outcomes and overall scores
- `src/xai_audit/graph.py`: LangGraph coordinator and staged persona workflow
- `src/xai_audit/simple_xai.py`: demo black-box model, fixed explainer, and demo validation set
- `src/xai_audit/technical.py`: optional technical metrics retained from the earlier prototype
- `src/xai_audit/benchmarks/datasets.py`: finance and healthcare benchmark dataset loaders
- `src/xai_audit/benchmarks/training.py`: neural baseline training and artifact export
- `scripts/train_benchmark_models.py`: CLI for fetching datasets, training models, and saving results
- `examples/run_demo.py`: runnable demonstration

## Core Assumptions

- The black-box model can be queried on one or more tabular rows.
- The fixed XAI method produces local explanations one sample at a time.
- The validation set contains at least 5 tabular rows.
- Sampled perturbation questions are generated from numeric features with strong local attribution magnitude.
- Understanding is operationalized as persona accuracy on perturbation questions.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run The Demo

The demo uses LangChain's unified chat-model initialization:

```bash
export OPENAI_API_KEY=...
export AUDIT_LLM_MODEL="openai:gpt-4.1-mini"
PYTHONPATH=src python3 examples/run_demo.py
```

The demo writes `audit_outcomes.csv` and prints its path.

The CSV contains:

- one row for each persona-stage outcome
- one row for each understanding-question interaction
- one row for the understanding observation interaction
- one overall row for each evaluated aspect: trust, usability, understanding

## Using Your Own Model And Explainer

Implement these two protocols:

- `TabularBlackBoxModel`
- `FixedTabularExplainer`

Then call `AuditCoordinator.audit(...)` with your validation set. If you want a file written automatically, pass `csv_output_path=...`.

## Notes

- Sample selection is deterministic and score-spread based.
- Trust, usability, and understanding are evaluated one at a time in separate coordinator-persona interactions.
- The understanding score is derived automatically from persona quiz accuracy, then mapped to a 1 to 5 scale.
- No LLM-based final report is generated; the final artifact is a deterministic CSV.
- The prototype is intentionally compact so you can swap in stronger explainers, richer questionnaires, or more sophisticated sample selection later.

## Benchmark Datasets

The repository now includes four tabular classification benchmarks for assessing the framework in finance and healthcare:

- `german_credit` (`finance`): UCI Statlog German Credit Data
- `credit_card_default` (`finance`): UCI Default of Credit Card Clients
- `heart_disease` (`healthcare`): UCI Heart Disease
- `breast_cancer` (`healthcare`): scikit-learn Wisconsin Diagnostic Breast Cancer

Each benchmark is paired with a PyTorch MLP baseline for tabular binary classification. The preprocessing pipeline is saved separately from the neural network checkpoint.

## Train Benchmark Models

After installing dependencies, run:

```bash
PYTHONPATH=src python3 scripts/train_benchmark_models.py --datasets all
```

Artifacts are saved under `artifacts/benchmark_models/<dataset_name>/` and include:

- `model.pt`
- `preprocessor.joblib`
- `metrics.json`
- `metadata.json`
- `dataset.csv`
- `test_split.csv`
- `training_summary.json`

## Dataset Sources

- German Credit: [UCI Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- Credit Card Default: [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- Heart Disease: [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- Breast Cancer: [scikit-learn load_breast_cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
