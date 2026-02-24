"""Email tone classifier with Langfuse observability.

This module demonstrates:
- Langfuse trace creation and span nesting
- Generation tracking (prompt, completion, token usage, cost)
- Attaching scores to traces
- Tagging and metadata for filtering in the dashboard
"""

from __future__ import annotations

import re

import openai
from langfuse import Langfuse

from src.prompts import ALL_VARIANTS, PromptVariant

# Valid tone labels the classifier can return
VALID_TONES = frozenset({"formal", "casual", "urgent", "friendly", "angry"})

MODEL = "gpt-4o-mini"


class ToneClassifier:
    """Classifies email tone using OpenAI, with full Langfuse tracing."""

    def __init__(self, langfuse: Langfuse, client: openai.OpenAI) -> None:
        self.langfuse = langfuse
        self.client = client

    def classify(
        self,
        email: str,
        variant_name: str = "simple",
        *,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Classify an email's tone and trace the full pipeline in Langfuse.

        Args:
            email: The email text to classify.
            variant_name: Which prompt variant to use (see prompts.py).
            tags: Optional tags for filtering in Langfuse dashboard.
            metadata: Optional metadata to attach to the trace.

        Returns:
            Dict with keys: tone, raw_response, variant, trace_id.
        """
        variant = ALL_VARIANTS[variant_name]

        # -- Create a top-level Langfuse trace (v3: root span = trace) --
        with self.langfuse.start_as_current_span(
            name="classify_email_tone",
            input={"email": email},
            metadata={"variant": variant_name, **(metadata or {})},
        ) as root_span:
            root_span.update_trace(tags=tags or [])

            # -- Span for input preprocessing --
            preprocess_span = root_span.start_span(
                name="preprocess",
                input={"raw_email": email},
            )
            cleaned = email.strip()
            preprocess_span.update(output={"cleaned_email": cleaned, "char_count": len(cleaned)})
            preprocess_span.end()

            # -- Generation: the actual LLM call --
            user_content = variant.user_template.format(email=cleaned)

            generation = root_span.start_generation(
                name="llm_classification",
                model=MODEL,
                model_parameters={"temperature": "0.0", "max_tokens": 256},
                input=[
                    {"role": "system", "content": variant.system},
                    {"role": "user", "content": user_content},
                ],
                metadata={"variant": variant_name},
            )

            response = self.client.chat.completions.create(
                model=MODEL,
                max_tokens=256,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": variant.system},
                    {"role": "user", "content": user_content},
                ],
            )

            raw_text = response.choices[0].message.content
            usage = response.usage

            generation.update(
                output=raw_text,
                usage_details={
                    "input": usage.prompt_tokens,
                    "output": usage.completion_tokens,
                    "total": usage.total_tokens,
                },
            )
            generation.end()

            # -- Span for output parsing --
            parse_span = root_span.start_span(
                name="parse_tone",
                input={"raw_response": raw_text},
            )
            tone = self._extract_tone(raw_text, variant)
            is_valid = tone in VALID_TONES
            parse_span.update(output={"tone": tone, "valid": is_valid})
            parse_span.end()

            # -- Score the trace with a confidence metric --
            root_span.score_trace(
                name="parse_success",
                value=1.0 if is_valid else 0.0,
                comment=f"Extracted tone: {tone}" if is_valid else "Failed to parse tone",
            )

            trace_id = self.langfuse.get_current_trace_id()

        return {
            "tone": tone,
            "raw_response": raw_text,
            "variant": variant_name,
            "trace_id": trace_id,
        }

    @staticmethod
    def _extract_tone(raw_text: str, variant: PromptVariant) -> str:
        """Pull the tone label out of the LLM response."""
        text_lower = raw_text.strip().lower()

        # For chain-of-thought, look for "TONE: <label>" pattern
        if variant.name == "chain_of_thought":
            match = re.search(r"tone:\s*(\w+)", text_lower)
            if match:
                return match.group(1)

        # For simple/few-shot, the response should just be the label
        first_word = text_lower.split()[0].strip(".,!") if text_lower else ""
        if first_word in VALID_TONES:
            return first_word

        # Fallback: scan for any valid tone in the response
        for tone in VALID_TONES:
            if tone in text_lower:
                return tone

        return "unknown"
