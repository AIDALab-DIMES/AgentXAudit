from __future__ import annotations

import json
import os
import sys

from langchain.chat_models import init_chat_model

from xai_audit.graph import AuditCoordinator
from xai_audit.simple_xai import (
    FiniteDifferenceTabularExplainer,
    ToyCreditRiskModel,
    build_demo_validation_set,
)


def main() -> int:
    model_spec = os.environ.get("AUDIT_LLM_MODEL", "openai:gpt-4.1-mini")

    try:
        llm = init_chat_model(model_spec)
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
    validation_set = build_demo_validation_set()

    reference_row = {
        "income_k": 55.0,
        "debt_ratio": 0.35,
        "savings_k": 12.0,
        "missed_payments": 1.0,
        "age": 40.0,
    }
    user_query = (
        "Audit whether these validation-set explanations are trustworthy, usable, and understandable "
        "for internal reviewers and for a customer affected by the credit decision."
    )

    result = coordinator.audit(
        user_query=user_query,
        validation_set=validation_set,
        model=predictor,
        explainer=explainer,
        reference_row=reference_row,
        focus_label="approve",
        sample_count=5,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
