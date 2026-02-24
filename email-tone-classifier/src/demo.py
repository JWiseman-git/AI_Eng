"""Interactive demo — run all prompt variants and trace them in Langfuse.

Usage:
    uv run python -m src.demo
"""

from __future__ import annotations

import openai
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
    # -- Hard / likely-to-misclassify examples --
    {
        # Passive-aggressive politeness — sounds formal but the intent is angry.
        # Classifiers often label this "formal" instead of "angry".
        "email": (
            "Dear Support Team, I trust you are well. I am once again reaching out "
            "regarding the unresolved billing discrepancy from last quarter. I have "
            "now sent four emails on this matter without a single substantive reply. "
            "I would appreciate your earliest attention. Kindest regards, Marcus."
        ),
        "expected": "angry",
    },
    {
        # Casual register + emoji, but the situation is a live outage — urgent.
        # Classifiers often latch onto the casual style and miss the urgency.
        "email": (
            "hey team 👋 so uh… the prod DB just went read-only like 10 mins ago "
            "and orders are failing 😬 someone pls take a look asap?? thx"
        ),
        "expected": "urgent",
    },
    {
        # Warm, friendly opener wrapping a formal request letter.
        # Classifiers often return "friendly" instead of "formal".
        "email": (
            "Hi Sarah! It was wonderful catching up at the summit last week. "
            "I am writing to formally submit our department's budget proposal for "
            "FY 2026, as outlined in the attached documentation. Please confirm "
            "receipt at your earliest convenience. Warm regards, Priya."
        ),
        "expected": "formal",
    },
    {
        # Dripping sarcasm — reads superficially as friendly/positive praise.
        # Classifiers frequently return "friendly" instead of "angry".
        "email": (
            "Wow, another missed deadline — truly impressive! I guess waiting six "
            "weeks for a two-page report is just how we do things here now. "
            "Fantastic work, everyone. Really setting the bar high."
        ),
        "expected": "angry",
    },
    {
        # Formal vocabulary but the message is a casual social invitation.
        # Classifiers often return "formal" instead of "casual".
        "email": (
            "Dear Jordan, I am reaching out to inquire as to whether you would be "
            "amenable to joining us for a small gathering at my residence this "
            "Saturday evening. Nothing too elaborate — just a few friends, some "
            "drinks, and good conversation. Do let me know! Cheers, Oliver."
        ),
        "expected": "casual",
    },
]


def run_demo() -> None:
    """Classify all sample emails with every prompt variant."""
    langfuse = Langfuse()
    client = openai.OpenAI()
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
