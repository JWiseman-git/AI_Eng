"""Interactive demo — run all prompt variants and trace them in Langfuse.

Usage:
    uv run python -m src.demo
"""

from __future__ import annotations

import anthropic
from dotenv import load_dotenv
from langfuse import Langfuse

from src.classifier import ToneClassifier
from src.prompts import ALL_VARIANTS

load_dotenv()

# -- Sample emails covering each tone --
SAMPLE_EMAILS = [
    {
        "email": (
            "Dear Dr. Patel, I am writing to formally request an extension "
            "on the project deadline. Please let me know your availability "
            "for a meeting this week. Best regards, Jordan."
        ),
        "expected": "formal",
    },
    {
        "email": "hey! saw that new café on st-denis, wanna check it out tmrw? 🙂",
        "expected": "casual",
    },
    {
        "email": (
            "CRITICAL: The deployment pipeline is broken and customers are "
            "seeing 500 errors. We need all hands on deck NOW. Drop everything."
        ),
        "expected": "urgent",
    },
    {
        "email": (
            "Just wanted to say thanks for helping me debug that issue yesterday. "
            "You really saved my day! Looking forward to working together more."
        ),
        "expected": "friendly",
    },
    {
        "email": (
            "I have been waiting three weeks for a response. This is completely "
            "unacceptable and I am considering escalating this to your manager. "
            "Fix this now."
        ),
        "expected": "angry",
    },
]


def run_demo() -> None:
    """Classify all sample emails with every prompt variant."""
    langfuse = Langfuse()
    client = anthropic.Anthropic()
    classifier = ToneClassifier(langfuse=langfuse, client=client)

    print("=" * 60)
    print("Email Tone Classifier — Langfuse Tracing Demo")
    print("=" * 60)

    results: list[dict] = []

    for variant_name in ALL_VARIANTS:
        print(f"\n--- Variant: {variant_name} ---")

        for sample in SAMPLE_EMAILS:
            result = classifier.classify(
                email=sample["email"],
                variant_name=variant_name,
                tags=["demo", f"variant:{variant_name}"],
                metadata={"expected_tone": sample["expected"]},
            )

            match = "✓" if result["tone"] == sample["expected"] else "✗"
            print(
                f"  {match} Expected: {sample['expected']:8s} "
                f"Got: {result['tone']:8s} "
                f"(trace: {result['trace_id'][:12]}...)"
            )

            results.append(
                {
                    **result,
                    "expected": sample["expected"],
                    "correct": result["tone"] == sample["expected"],
                }
            )

    # -- Summary --
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for variant_name in ALL_VARIANTS:
        variant_results = [r for r in results if r["variant"] == variant_name]
        correct = sum(1 for r in variant_results if r["correct"])
        total = len(variant_results)
        print(f"  {variant_name:20s} {correct}/{total} correct")

    # Flush all pending Langfuse events
    langfuse.flush()
    print("\n✓ All traces sent to Langfuse. Check your dashboard!")


if __name__ == "__main__":
    run_demo()
