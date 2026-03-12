"""Minimal CLI demo for the LangGraph farm advisory workflow."""
from agent import run_advisory


def main() -> None:
    result = run_advisory(
        {
            "crop": "Maize",
            "area": "Kenya",
            "year": 2005,
            "rainfall": 900.0,
            "pesticides": 500.0,
            "temperature": 22.0,
        }
    )
    print("status:", result["status"])
    print("predicted_yield:", result["predicted_yield"], "category:", result["yield_category"])


if __name__ == "__main__":
    main()
