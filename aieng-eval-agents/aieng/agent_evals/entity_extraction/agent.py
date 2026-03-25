"""Entity extraction agent.

This module defines the factory used to build the entity extraction agent,
and ``run_entity_extraction`` for one-shot async execution (e.g. CLI).

The returned agent is a Google ADK ``LlmAgent`` configured to:

- Accept article text (title + maintext) as input.
- Extract named entities and mentioned companies.
- Use Google Search when needed to resolve ticker symbols for company names.
- Emit JSON parsed and validated as ``EntityExtractionOutput`` (no ADK
  ``output_schema`` alongside tools: nested Pydantic schemas use JSON Schema
  ``$ref``/``$defs``, which Google's tool declaration format does not accept).

Examples
--------
>>> from aieng.agent_evals.entity_extraction.agent import create_entity_extraction_agent
>>> agent = create_entity_extraction_agent()
>>> agent.name
'EntityExtractionAgent'
"""

import json
import re
import uuid

from aieng.agent_evals.configs import Configs
from aieng.agent_evals.entity_extraction.entity_extraction_models import EntityExtractionOutput
from aieng.agent_evals.tools import create_google_search_tool
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.genai.types import GenerateContentConfig, ThinkingConfig


_DEFAULT_AGENT_DESCRIPTION = (
    "Extracts named entities and mentioned companies from article text."
)

EXTRACTION_PROMPT = """\
You are a named-entity extraction specialist. Your task is to read the provided \
article (title + maintext) and extract every named entity, as well as any \
company ticker symbols that are explicitly mentioned or clearly identifiable \
from the text.

## Input

You will receive a JSON object with two fields:
- `title`: the article headline.
- `maintext`: the full article body.

## Extraction Rules

### mentioned_companies
- List the **ticker symbols** of every company that is explicitly mentioned \
  in the text, whose ticker symbol appears directly in the text, or whose ticker \
  you resolve using ``google_search`` as described below.

### Tool: google_search (required for tradable companies)
- You **must** call ``google_search(query)`` at least once for each **distinct** \
  ``ORG`` that names a **publicly traded company or major consumer brand** when \
  the article does **not** already state its ticker. Do **not** skip search for \
  brands like device makers, streaming services, or tech platforms (e.g. Fitbit, \
  Apple, Beats, Pandora, MarketWatch) unless you already have a correct ticker \
  from the article text.
- **Skip** search for obvious non‑issuers: universities, labs, academic journals, \
  cities, courts, regulators, and individual people (even if ``ORG``-like titles).
- Each query should be focused, e.g. ``"{company name}" stock ticker`` or \
  ``"{company name}" NYSE NASDAQ symbol``.
- Use the ``summary`` (and titles in ``sources``) to pick **one** ticker; add it \
  to ``mentioned_companies`` and set ``normalized`` on that entity’s row only \
  when results clearly support a single symbol. If ambiguous or not listed, \
  leave ``normalized`` null—**do not** invent one‑letter symbols from the \
  company’s initial (e.g. do not map “Pandora” to ``P`` without clear evidence).
- ``normalized`` may come from the article **or** from search results; both are \
  allowed.

### named_entities
For every named entity found in the title or maintext, produce an object with:

| Field          | Description |
|----------------|-------------|
| `entity_group` | One of `ORG` (organisation), `PER` (person), `LOC` (location), `MISC` (miscellaneous). |
| `word`         | The entity text **exactly as it appears** in the source. |
| `normalized`   | Ticker or canonical form from the **article text** or **verified via** ``google_search``. Use `null` when unknown or not a listed company. |

### Guidelines
- Extract **all** entities, not just the most prominent ones.
- Preserve the original surface form in `word`; do not alter capitalisation or spelling.
- Classify courts, government bodies, and regulatory agencies as `ORG`.
- Classify countries, cities, continents, and regions as `LOC`.
- When the same entity appears multiple times, include it **only once**.
- Do not fabricate entities that are not in the text.

## Output
After you finish any ``google_search`` calls, respond with **only** a single JSON \
object and no other prose or markdown. Shape:

- ``mentioned_companies``: array of strings (ticker symbols).
- ``named_entities``: array of objects, each with ``entity_group`` (``ORG``, \
  ``PER``, ``LOC``, or ``MISC``), ``word`` (string), and ``normalized`` \
  (string or null).
"""


def create_entity_extraction_agent(
    name: str = "EntityExtractionAgent",
    *,
    description: str | None = None,
    instructions: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    thinking_budget: int = 2048,
) -> LlmAgent:
    """Create a configured entity extraction agent.

    Parameters
    ----------
    name : str, default="EntityExtractionAgent"
        Name assigned to the agent.
    description : str | None, optional
        Short description of the agent's purpose.
    instructions : str | None, optional
        System prompt override. Falls back to ``EXTRACTION_PROMPT``.
    model : str | None, optional
        Model name override. Falls back to ``Configs.default_worker_model``.
    temperature : float | None, optional
        Sampling temperature.
    thinking_budget : int, default 2048
        Token budget for the model's thinking phase.

    Returns
    -------
    LlmAgent
        Configured entity extraction agent with Google Search. Final JSON is
        validated by ``run_entity_extraction`` into ``EntityExtractionOutput``.
    """
    config = Configs()  # type: ignore[call-arg]
    search_tool = create_google_search_tool(config)
    resolved_model = model or config.default_worker_model

    thinking_config = None
    if thinking_budget > 0:
        model_lower = resolved_model.lower()
        if "gemini-2.5" in model_lower or "gemini-3" in model_lower:
            thinking_config = ThinkingConfig(thinking_budget=thinking_budget)

    return LlmAgent(
        name=name,
        description=description or _DEFAULT_AGENT_DESCRIPTION,
        model=resolved_model,
        instruction=instructions or EXTRACTION_PROMPT,
        tools=[search_tool],
        generate_content_config=GenerateContentConfig(
            temperature=temperature,
            thinking_config=thinking_config,
        ),
    )


def _coerce_final_json_text(raw: str) -> str:
    """Strip optional markdown fences so model output still validates."""
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*\r?\n?(.*)\r?\n?```\s*$", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text


def _final_response_text_from_event(event: object) -> str | None:
    """Text from a final model event, skipping thought/reasoning parts (Gemini thinking)."""
    content = getattr(event, "content", None)
    if not content or not getattr(content, "parts", None):
        return None
    parts: list[str] = []
    for part in content.parts:
        if getattr(part, "thought", False):
            continue
        if hasattr(part, "text") and part.text:
            parts.append(part.text)
    joined = "\n".join(parts).strip()
    return joined if joined else None


def _first_json_object_slice(text: str) -> str | None:
    """If the model wrapped JSON in prose, return the first balanced `{...}` substring."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_entity_output(raw: str) -> EntityExtractionOutput:
    """Parse model output into ``EntityExtractionOutput`` with several fallbacks."""
    coerced = _coerce_final_json_text(raw)
    if not coerced.strip():
        raise ValueError("Model output is empty after stripping fences.")

    try:
        return EntityExtractionOutput.model_validate_json(coerced)
    except Exception:
        pass

    try:
        return EntityExtractionOutput.model_validate(json.loads(coerced))
    except Exception:
        pass

    slice_json = _first_json_object_slice(coerced)
    if slice_json:
        try:
            return EntityExtractionOutput.model_validate_json(slice_json)
        except Exception:
            return EntityExtractionOutput.model_validate(json.loads(slice_json))

    raise ValueError(
        "Could not parse entity extraction JSON from model output. "
        f"First 500 chars: {raw[:500]!r}"
    )


async def run_entity_extraction(title: str, maintext: str) -> EntityExtractionOutput:
    """Run the entity extraction agent on one article and return structured output."""
    agent = create_entity_extraction_agent()
    session_service = InMemorySessionService()
    runner = Runner(
        app_name=agent.name,
        agent=agent,
        session_service=session_service,
        auto_create_session=True,
    )
    try:
        payload = json.dumps({"title": title, "maintext": maintext}, ensure_ascii=False)
        message = types.Content(parts=[types.Part(text=payload)], role="user")
        final_text: str | None = None
        async for event in runner.run_async(
            session_id=str(uuid.uuid4()),
            user_id="entity_extraction",
            new_message=message,
        ):
            if not event.is_final_response():
                continue
            chunk = _final_response_text_from_event(event)
            # ADK may emit multiple "final" events; later ones can be empty and must not
            # overwrite a valid JSON response from an earlier turn.
            if chunk:
                final_text = chunk

        if not final_text or not final_text.strip():
            raise RuntimeError("Entity extraction produced no output.")

        return _parse_entity_output(final_text)
    finally:
        await runner.close()
