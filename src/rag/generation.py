from __future__ import annotations
import os
from .chunking import Chunk

SYSTEM_PROMPT = """\
You are a credit risk explanation assistant for a bank's loan officers.

You will be given:
1. A machine learning model's predicted probability of default (PD) for an applicant,
   and the top factors (SHAP values) that drove that prediction.
2. Excerpts retrieved from the bank's official credit, lending, approval, risk,
   collateral, and exception policies.

Your job is to explain the applicant's risk assessment and how it relates to bank
policy, USING ONLY the model output and the retrieved policy excerpts provided to
you. Follow these rules strictly:

- Do NOT invent financial rules, thresholds, or policy language that isn't in the
  retrieved excerpts. If the excerpts don't cover something, say so explicitly
  rather than guessing.
- Do NOT invent or restate risk factors beyond what the SHAP output provided —
  you are explaining the model's evidence, not performing your own risk analysis.
- Every policy claim in your answer must be traceable to a specific retrieved
  excerpt. Reference excerpts by their source document name.
- If the retrieved excerpts are insufficient to answer the question, say that
  plainly instead of filling the gap yourself.
- Keep the tone professional and factual, appropriate for a loan officer's file.
"""

USER_PROMPT_TEMPLATE = """\
## Applicant risk assessment
Predicted probability of default: {pd_score:.1%}
Top risk drivers (from SHAP):
{risk_drivers}

## Retrieved policy excerpts
{policy_excerpts}

## Question
{question}
"""


def format_risk_drivers(drivers: dict[str, float]) -> str:
    lines = [f"- {name}: {value:+.4f}" for name, value in drivers.items()]
    return "\n".join(lines) if lines else "(none provided)"


def format_policy_excerpts(chunks: list[Chunk]) -> str:
    if not chunks:
        return "(no relevant policy excerpts retrieved)"
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{i}] Source: {chunk.source}\n{chunk.text}")
    return "\n\n".join(blocks)


def build_prompt(
    question: str,
    pd_score: float,
    risk_drivers: dict[str, float],
    retrieved_chunks: list[Chunk],
) -> tuple[str, str]:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        pd_score=pd_score,
        risk_drivers=format_risk_drivers(risk_drivers),
        policy_excerpts=format_policy_excerpts(retrieved_chunks),
        question=question,
    )
    return SYSTEM_PROMPT, user_prompt


def generate_answer(
    question: str,
    pd_score: float,
    risk_drivers: dict[str, float],
    retrieved_chunks: list[Chunk],
    model: str = "claude-sonnet-4-6",
) -> str:
    import anthropic

    system_prompt, user_prompt = build_prompt(question, pd_score, risk_drivers, retrieved_chunks)

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")