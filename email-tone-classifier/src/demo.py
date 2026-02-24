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

    # -- Long-form examples (produce larger traces) --
    {
        # Multi-paragraph formal legal/compliance notice.
        # Length and dense structure stress-test the chain-of-thought variant.
        "email": (
            "Dear Mr. Henderson,\n\n"
            "I am writing on behalf of Meridian Financial Services Ltd. to formally "
            "notify you of a material change to the terms and conditions governing "
            "your account, effective 1 April 2026. This communication constitutes "
            "the required thirty-day written notice as stipulated in Section 14(b) "
            "of your account agreement dated 12 March 2023.\n\n"
            "Specifically, the following amendments will take effect: (1) the annual "
            "maintenance fee will increase from £24.00 to £36.00; (2) the minimum "
            "balance required to waive said fee will rise from £1,000 to £2,500; "
            "and (3) international wire transfer charges will be revised from a flat "
            "£15 to 0.75% of the transferred amount, subject to a minimum of £10 "
            "and a maximum of £60.\n\n"
            "Should you wish to dispute these changes or close your account without "
            "penalty before the effective date, please contact our Client Relations "
            "team in writing at the address below, or via our secure online portal. "
            "We are required by the Financial Conduct Authority to retain a copy of "
            "all such correspondence for a period of not less than five years.\n\n"
            "Please review the enclosed schedule of revised fees in its entirety. "
            "Your continued use of the account after 1 April 2026 will constitute "
            "acceptance of the updated terms. We regret any inconvenience this may "
            "cause and thank you for your continued custom.\n\n"
            "Yours sincerely,\nAmelia Frost\nHead of Client Compliance, Meridian FS"
        ),
        "expected": "formal",
    },
    {
        # Long, escalating customer complaint — starts politely, ends furious.
        # The shift in register across paragraphs is hard for classifiers to weight.
        "email": (
            "Hi there,\n\n"
            "I placed order #TK-88271 on 3 January and was quoted a delivery window "
            "of 5–7 business days. It is now 28 February — nearly two months later — "
            "and I still do not have my order.\n\n"
            "I have contacted your support team six times. The first three times I "
            "was told the parcel was 'in transit'. The fourth time I was told it had "
            "been 'lost in the warehouse' and a replacement would be dispatched within "
            "48 hours. That was three weeks ago. The fifth time the agent simply "
            "disconnected the chat. The sixth time I was placed on hold for 47 minutes "
            "before being told someone would 'call me back within 24 hours'. That was "
            "eleven days ago. No one has called.\n\n"
            "I have now raised a chargeback with my bank and filed a complaint with "
            "Trading Standards. I am also posting a detailed account of this "
            "experience to every consumer review platform I can find. I want a full "
            "refund processed within 48 hours — not a voucher, not store credit, not "
            "a replacement. A refund. If that does not happen I will be pursuing this "
            "through the small claims court. I am done being patient.\n\n"
            "Do not send me another templated apology email."
        ),
        "expected": "angry",
    },
    {
        # Long, rambling group-chat-style message about an active production outage.
        # Casual language and humour mask genuine severity — tricky for classifiers.
        "email": (
            "ok so heads up everyone 🚨\n\n"
            "so i was just doing the routine Monday deploy (yes i know, i know — "
            "never deploy on Monday, lesson thoroughly learned lmao) and uh… "
            "things have gone a bit sideways. the payment service is throwing "
            "NullPointerExceptions on every third request, the checkout page is "
            "just showing a blank white screen for about 30% of users, and "
            "Datadog is absolutely losing its mind with alerts rn.\n\n"
            "i've already rolled back the last commit but the errors are still "
            "happening which means it's probably not the deploy itself — could be "
            "the new Stripe webhook config we pushed last Thursday? or possibly "
            "the Redis cache eviction policy change? idk 😬\n\n"
            "revenue is definitely being impacted — i can see abandoned carts "
            "spiking hard in the analytics dashboard. we're probably losing "
            "somewhere between £200-£500 per minute at current traffic levels.\n\n"
            "can we get ALL hands on this please? like right now? "
            "@backend-team can someone dig into the payment service logs? "
            "@devops can you check if there's anything weird in infra? "
            "i'm going to start a war-room call in 5 mins, link in #incidents 👇"
        ),
        "expected": "urgent",
    },
    {
        # Long, heartfelt team appreciation message with lots of personal detail.
        # Pure friendly tone sustained across many sentences — good positive example.
        "email": (
            "Hey everyone,\n\n"
            "I just wanted to take a moment — and yes I know we're all slammed "
            "right now — to say how genuinely grateful I am to be part of this team.\n\n"
            "The last three months have been a lot. The product pivot, the two "
            "all-hands firefights, the late nights before the investor demo… none "
            "of that was easy, and honestly I wasn't sure we were going to pull it "
            "off. But we did, and it's entirely because of the people in this room "
            "(or, well, in this Slack channel).\n\n"
            "Special shout-out to Kenji for completely rewriting the data pipeline "
            "over a single weekend without being asked — that unblocked the entire "
            "analytics team for two weeks. And to Rosa, who caught that silent data "
            "corruption bug at 11pm on a Friday and stayed until 2am to fix it "
            "rather than leaving it for Monday. That kind of care is rare.\n\n"
            "I also just want to say: if anyone is feeling burnt out, please tell "
            "me. We're going to celebrate properly next week — dinner on the company "
            "card, no laptops, no Slack. You've all more than earned it.\n\n"
            "Seriously, thank you. I'm lucky to work with you.\n\nTom"
        ),
        "expected": "friendly",
    },
    {
        # Long formal academic/research collaboration proposal.
        # Dense, structured, impersonal — strong formal signal throughout.
        "email": (
            "Dear Professor Nakamura,\n\n"
            "I am writing to formally propose a research collaboration between the "
            "Computational Linguistics Laboratory at the University of Edinburgh and "
            "your Natural Language Processing group at Kyoto University, with the "
            "aim of jointly submitting a grant application to the UKRI–JST bilateral "
            "research fund, deadline 30 June 2026.\n\n"
            "Our proposed project, provisionally titled 'Cross-Lingual Pragmatic "
            "Inference in Low-Resource Settings', would build upon your group's "
            "published work on Japanese discourse markers (Nakamura et al., 2024) "
            "and our recent results in zero-shot transfer for pragmatic tasks "
            "(Singh & O'Brien, 2025). We believe the complementary nature of our "
            "methodological approaches — your group's expertise in linguistically "
            "motivated feature engineering and our team's focus on large-scale "
            "pre-training — presents a strong basis for a productive partnership.\n\n"
            "The collaboration would involve a twelve-month exchange programme for "
            "two doctoral researchers (one from each institution), four joint "
            "workshops to be held alternately in Edinburgh and Kyoto, and a shared "
            "annotated corpus to be released under a Creative Commons licence upon "
            "project completion.\n\n"
            "I would welcome the opportunity to arrange a video conference at your "
            "earliest convenience to discuss the proposal in greater detail. Please "
            "find attached a two-page concept note for your review. I look forward "
            "to your response.\n\n"
            "Yours sincerely,\nDr. Ananya Singh\n"
            "Reader in Computational Linguistics, University of Edinburgh"
        ),
        "expected": "formal",
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
