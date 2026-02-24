# Email Tone Classifier — Langfuse + Promptfoo Workshop

A hands-on ~2 hour project to learn **Langfuse** (LLM observability/tracing) and
**Promptfoo** (prompt evaluation) using the Anthropic API.

## What You'll Learn

| Tool | Concepts |
|------|----------|
| **Langfuse** | Traces, spans, generations, prompt management, scoring, cost tracking |
| **Promptfoo** | Prompt variants, test datasets, assertion-based evaluation, grading |
| **Anthropic** | Claude messages API, structured output, system prompts |

## Project Structure

```
email-tone-classifier/
├── src/
│   ├── __init__.py
│   ├── classifier.py        # Core classifier with Langfuse tracing
│   ├── prompts.py            # Prompt variants to compare
│   └── demo.py               # Interactive demo with Langfuse traces
├── eval/
│   ├── promptfooconfig.yaml  # Promptfoo evaluation config
│   └── dataset.yaml          # Test cases for evaluation
├── pyproject.toml
├── .env.example
└── README.md
```

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Anthropic API key
- Langfuse account (free tier at https://cloud.langfuse.com)
- Node.js 18+ (for Promptfoo CLI)

## Setup (~15 min)

### 1. Create environment and install dependencies

```bash
cd email-tone-classifier
uv sync
```

### 2. Set up Langfuse

1. Sign up at https://cloud.langfuse.com (free tier is fine)
2. Create a new project called "email-tone-classifier"
3. Go to Settings → API Keys → Create new key
4. Copy the public key, secret key, and host URL

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your keys
```

### 4. Install Promptfoo CLI

```bash
npm install -g promptfoo
```

## Part 1: Langfuse Tracing (~45 min)

Run the demo to send traced requests through Langfuse:

```bash
uv run python -m src.demo
```

Then open your Langfuse dashboard to explore:
- **Traces** — each classification is a full trace
- **Generations** — see the exact prompt/completion for each LLM call
- **Cost tracking** — token usage and estimated cost per call
- **Scores** — confidence scores attached to each trace

### Things to try
- Compare how different prompt variants appear in the Langfuse trace view
- Look at latency differences between prompt strategies
- Check the "Prompts" tab to see managed prompt versions

## Part 2: Promptfoo Evaluation (~45 min)

Run the evaluation suite to compare prompt variants:

```bash
cd eval
promptfoo eval
promptfoo view
```

This opens a local web UI showing:
- Side-by-side comparison of all prompt variants
- Pass/fail results for each test case
- Which prompt strategy wins overall

### Things to try
- Add more test cases to `dataset.yaml`
- Add a new prompt variant in `promptfooconfig.yaml`
- Try stricter assertions (exact match vs contains)
- Experiment with temperature settings

## Part 3: Combine Both (~30 min)

The Promptfoo config is already set up to send traces to Langfuse via the
custom provider in `src/classifier.py`. After running `promptfoo eval`, check
Langfuse to see all evaluation traces tagged with `source: promptfoo`.

## Key Takeaways

- **Langfuse** = production observability. Use it to monitor, debug, and
  track costs of your LLM calls in real applications.
- **Promptfoo** = development-time evaluation. Use it to systematically
  compare prompt variants before deploying changes.
- They complement each other: Promptfoo tells you *which* prompt is best,
  Langfuse tells you *how* it performs in production.
