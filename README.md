# XAI Audit Prototype

This project implements a prototype framework for auditing the explanations emitted by a fixed explainability method on tabular data.

The design follows the interaction schema you provided:

1. An audit coordinator initializes the shared audit state.
2. Three simulated stakeholder persona agents query the same LLM through role-specific prompts:
   - data scientist
   - business executive
   - end user
3. A technical assessor computes code-based metrics, currently:
   - reconstruction fidelity
   - directional faithfulness
   - stability under local perturbations
4. The coordinator synthesizes a final report.

The orchestration layer uses `LangGraph`, while LLM interactions use `LangChain`.

## Project Layout

- `src/xai_audit/interfaces.py`: model and explainer protocols
- `src/xai_audit/model_utils.py`: helpers for turning model outputs into auditable scores
- `src/xai_audit/technical.py`: code-based technical metrics
- `src/xai_audit/simple_xai.py`: a fixed black-box tabular explainer used in the demo
- `src/xai_audit/graph.py`: LangGraph coordinator and persona workflow
- `examples/run_demo.py`: runnable demonstration

## Core Assumptions

- The black-box model can be queried on one or more tabular rows.
- The fixed XAI method returns local feature attributions for one instance.
- The current prototype focuses on scalar tabular features. Numeric features are used for perturbation-based technical checks.
- Technical metrics are proxy audit signals, not formal guarantees.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run The Demo

The demo uses LangChain's unified model initialization and defaults to an OpenAI-compatible model string:

```bash
export OPENAI_API_KEY=...
export AUDIT_LLM_MODEL="openai:gpt-4.1-mini"
PYTHONPATH=src python3 examples/run_demo.py
```

You can swap the provider-backed LLM by changing `AUDIT_LLM_MODEL`, as long as the corresponding LangChain integration package is installed.

## Using Your Own Model And Explainer

Implement these two protocols:

- `TabularBlackBoxModel`
- `FixedTabularExplainer`

Then pass them to `AuditCoordinator.audit(...)`.

## Notes

- The prototype is intentionally small and explainable.
- It is designed to be extended with stronger fidelity tests, additional personas, and richer report schemas.
