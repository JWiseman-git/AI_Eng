"""Prompt variants for email tone classification.

Each variant represents a different prompting strategy. Promptfoo will evaluate
these head-to-head, and Langfuse will trace each call for observability.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptVariant:
    """A named prompt strategy for tone classification."""

    name: str
    system: str
    user_template: str


# --- Variant 1: Simple direct instruction ---

SIMPLE = PromptVariant(
    name="simple",
    system="You are an email tone classifier.",
    user_template=(
        "Classify the tone of this email as exactly one of: "
        "formal, casual, urgent, friendly, angry.\n\n"
        "Email: {email}\n\n"
        "Respond with ONLY the tone label, nothing else."
    ),
)

# --- Variant 2: Chain-of-thought with structured output ---

CHAIN_OF_THOUGHT = PromptVariant(
    name="chain_of_thought",
    system=(
        "You are an expert email tone analyst. You always reason step by step "
        "before giving your final classification."
    ),
    user_template=(
        "Analyze the tone of the following email.\n\n"
        "Email: {email}\n\n"
        "Step 1: Identify key phrases that indicate tone.\n"
        "Step 2: Consider the overall sentiment.\n"
        "Step 3: Choose exactly one label from: "
        "formal, casual, urgent, friendly, angry.\n\n"
        "Format your response as:\n"
        "REASONING: <your analysis>\n"
        "TONE: <label>"
    ),
)

# --- Variant 3: Few-shot examples ---

FEW_SHOT = PromptVariant(
    name="few_shot",
    system="You are an email tone classifier. Learn from the examples below.",
    user_template=(
        "Examples:\n\n"
        'Email: "Dear Mr. Smith, Please find attached the Q3 report."\n'
        "Tone: formal\n\n"
        'Email: "Hey! Wanna grab lunch? 😊"\n'
        "Tone: casual\n\n"
        'Email: "NEED THIS FIXED BY EOD. Production is down."\n'
        "Tone: urgent\n\n"
        'Email: "Thanks so much for your help, you made my day!"\n'
        "Tone: friendly\n\n"
        'Email: "This is unacceptable. I want a refund immediately."\n'
        "Tone: angry\n\n"
        "Now classify this email:\n\n"
        'Email: "{email}"\n'
        "Tone:"
    ),
)

# Registry of all variants for easy iteration
ALL_VARIANTS: dict[str, PromptVariant] = {
    v.name: v for v in [SIMPLE, CHAIN_OF_THOUGHT, FEW_SHOT]
}
