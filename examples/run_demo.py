from __future__ import annotations

import json
import os
import sys

from langchain_ollama import OllamaLLM

from xai_audit.graph import AuditCoordinator
from xai_audit.simple_xai import FiniteDifferenceTabularExplainer, ToyCreditRiskModel


def main() -> int:
    model_spec = os.environ.get("AUDIT_LLM_MODEL", "gemma3")

    try:
        llm = OllamaLLM(model="gemma3:latest")
    except Exception as exc:  # pragma: no cover - demo bootstrap path
        print(
            "Failed to initialize the LangChain chat model.\n"
            f"Requested model: {model_spec}\n"
            f"Error: {exc}",
            file=sys.stderr,
        )
        return 1

    predictor = ToyCreditRiskModel()
    explainer = FiniteDifferenceTabularExplainer()
    coordinator = AuditCoordinator(llm=llm)

    reference_row = {
        "income_k": 55.0,
        "debt_ratio": 0.35,
        "savings_k": 12.0,
        "missed_payments": 1.0,
        "age": 40.0,
    }
    instance = {
        "income_k": 42.0,
        "debt_ratio": 0.48,
        "savings_k": 6.0,
        "missed_payments": 2.0,
        "age": 29.0,
    }
    user_query = (
        "Audit whether this explanation would be appropriate to show to internal reviewers "
        "and to the customer affected by the credit decision."
    )

    result = coordinator.audit(
        user_query=user_query,
        instance=instance,
        model=predictor,
        explainer=explainer,
        reference_row=reference_row,
        focus_label="approve",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
