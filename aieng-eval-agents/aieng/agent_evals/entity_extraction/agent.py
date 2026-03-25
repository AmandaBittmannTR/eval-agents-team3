"""Entity extraction agent.

This module defines the factory used to build the entity extraction agent,
and ``run_entity_extraction`` for one-shot async execution (e.g. CLI).

The returned agent is a Google ADK ``LlmAgent`` configured to:

- Accept article text (title + maintext) as input.
- Extract named entities and mentioned companies.
- Use Google Search when needed to resolve ticker symbols for company names.
- Enforce structured JSON via prompt-embedded schema with an explicit example
  and field rules (neither ADK ``output_schema`` nor
  ``response_mime_type="application/json"`` can be used alongside tools in the
  current Gemini API). Post-hoc parsing fallbacks are retained as a safety net.

Langfuse tracing
~~~~~~~~~~~~~~~~
``run_entity_extraction`` accepts ``langfuse_tracing=True`` to initialize
OpenTelemetry tracing via Langfuse before the ADK runner starts. If you build
a custom ``Runner`` from ``create_entity_extraction_agent()`` instead, call
``from aieng.agent_evals.langfuse import init_tracing; init_tracing()`` once
before the first ``run_async`` to enable tracing.

Examples
--------
>>> from aieng.agent_evals.entity_extraction.agent import create_entity_extraction_agent
>>> agent = create_entity_extraction_agent()
>>> agent.name
'EntityExtractionAgent'
"""

import json
import logging
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

logger = logging.getLogger(__name__)

_DEFAULT_AGENT_DESCRIPTION = (
    "Extracts named entities and mentioned companies from article text."
)

EXTRACTION_PROMPT = """\
You are a named-entity extraction specialist. Your task is to read the provided \
article (title + maintext) and extract named entities and company ticker symbols.

## Input

You will receive a JSON object with two fields:
- `title`: the article headline.
- `maintext`: the full article body.

## Entity Group Classification Rules

You MUST follow these classification rules exactly:

### ORG -- publicly traded companies ONLY
- Use `ORG` **exclusively** for well-known publicly traded companies \
(e.g. Apple, Google, Amazon, Netflix, Verizon, Wells Fargo) and for \
ticker symbols that appear literally in the text (e.g. AAPL, GOOG, VZ).
- Do **NOT** classify the following as `ORG`: universities, research labs, \
academic journals, courts, government bodies, regulatory agencies, private \
companies, or consumer brands that are not major publicly traded companies.

### PER -- named individuals
- Use `PER` for named people (e.g. "Tim Cook", "Colin Camerer", "Trump").

### LOC -- countries and major cities only
- Use `LOC` for countries (e.g. "Canada", "America") and major cities \
(e.g. "New York", "San Francisco", "Louisville", "Miami").
- Do **NOT** use `LOC` for U.S. state names, continents, or regions. \
Classify those as `MISC` instead (e.g. "Florida" -> MISC, "Alabama" -> MISC).

### MISC -- everything else
- `MISC` is the default category for all other named entities, including:
  - Product names (iPhone, Apple Watch Series 2, MacBook Pro)
  - Technology and platform names (Bluetooth, Android, iOS, Snapchat)
  - Consumer brands that are not major publicly traded companies \
(Kickstarter, Beats, Parkside)
  - Nationalities and demonyms (Chinese, American, Democratic)
  - U.S. state names and regions (Florida, Alabama, Silicon Valley)
  - Any other proper noun that does not fit ORG, PER, or LOC

## Extraction Scope

Extract only clearly identifiable named entities. **Skip** the following:
- Universities, research labs, and academic institutions
- Courts, government agencies, and regulatory bodies
- Academic journals and publications
- Generic descriptors, adjectives, dates, and fragmentary text
- Color names, model numbers in isolation, and minor references

Focus on: publicly traded companies, named individuals, key geographic \
references, and prominent product/brand/technology names.

## mentioned_companies

- List the **ticker symbols** only for entities you classified as `ORG`.
- Include a ticker if it appears explicitly in the article text (e.g. in \
parentheses like ``(AAPL)``), or if you resolve it via ``google_search``.
- Only include tickers for companies that are well-known and clearly \
identifiable as publicly traded.

## Tool-Calling Protocol (google_search)

Follow these steps **in order**:

1. **Scan** the article and identify every distinct `ORG` entity (publicly \
traded companies only, per the rules above).
2. **For each ORG whose ticker does NOT already appear in the article text**, \
call ``google_search(query)`` to resolve its ticker.
3. **After ALL searches are complete**, produce your final JSON output.

### Query format examples
- ``"Apple Inc" stock ticker symbol`` -> expect AAPL
- ``"Wells Fargo" NYSE ticker`` -> expect WFC
- ``"Netflix" NASDAQ ticker symbol`` -> expect NFLX

### Interpreting search results
- Use the ``summary`` and ``sources`` titles to identify **one** unambiguous \
ticker symbol.
- If the results are ambiguous or the company is not publicly listed, set \
``normalized`` to ``null`` and do NOT add it to ``mentioned_companies``.
- Do **NOT** invent ticker symbols.

### Do NOT search for
- Universities, research labs, academic journals
- Cities, countries, geographic regions
- Courts, regulators, government agencies
- Individual people
- Consumer brands that are not major publicly traded companies

### Critical constraints
- Complete ALL ``google_search`` calls BEFORE producing your final JSON.
- Do NOT output partial JSON between tool calls.

## named_entities format

For each entity, produce an object with:

| Field          | Description |
|----------------|-------------|
| `entity_group` | One of `ORG`, `PER`, `LOC`, `MISC` per the rules above. |
| `word`         | The entity text **exactly as it appears** in the source. |
| `normalized`   | Ticker symbol for `ORG` entities (from text or search). Use `null` for all other entities. |

### Additional guidelines
- Preserve the original surface form in `word`; do not alter capitalisation.
- When the same entity appears multiple times, include it **only once**.
- Do not fabricate entities that are not in the text.

## Output

Your final response must be a **single JSON object** with exactly this structure \
(no other prose, markdown fences, or commentary):

```
{
  "mentioned_companies": ["AAPL", "GOOGL"],
  "named_entities": [
    {"entity_group": "ORG", "word": "Apple", "normalized": "AAPL"},
    {"entity_group": "ORG", "word": "AAPL", "normalized": "AAPL"},
    {"entity_group": "PER", "word": "Tim Cook", "normalized": null},
    {"entity_group": "LOC", "word": "New York", "normalized": null},
    {"entity_group": "MISC", "word": "iPhone", "normalized": null},
    {"entity_group": "MISC", "word": "Florida", "normalized": null},
    {"entity_group": "MISC", "word": "American", "normalized": null}
  ]
}
```

Field rules:
- ``mentioned_companies``: array of strings (ticker symbols for ORG entities only).
- ``named_entities``: array of objects, each with:
  - ``entity_group``: one of ``"ORG"``, ``"PER"``, ``"LOC"``, ``"MISC"`` (no other values).
  - ``word``: string, exact surface form from the article.
  - ``normalized``: string (ticker symbol) for ORG entities, or ``null`` for all others.
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
        Configured entity extraction agent with Google Search. JSON structure
        is enforced via prompt-embedded schema and example. Post-hoc parsing
        fallbacks in ``run_entity_extraction`` provide defence-in-depth.
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
        logger.warning("Model output contained markdown fences despite JSON enforcement.")
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
    """If the model wrapped JSON in prose, return the first balanced ``{...}`` substring."""
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
    """Parse model output into ``EntityExtractionOutput`` with several fallbacks.

    The primary ``model_validate_json`` path should succeed when the model
    follows the prompt-embedded schema. The subsequent fallbacks are retained
    as defence-in-depth and emit warnings when they activate.
    """
    coerced = _coerce_final_json_text(raw)
    if not coerced.strip():
        raise ValueError("Model output is empty after stripping fences.")

    try:
        return EntityExtractionOutput.model_validate_json(coerced)
    except Exception:
        logger.warning(
            "Primary JSON parse failed; attempting json.loads fallback. "
            "First 200 chars: %s",
            coerced[:200],
        )

    try:
        return EntityExtractionOutput.model_validate(json.loads(coerced))
    except Exception:
        logger.warning("json.loads fallback also failed; trying JSON object slice extraction.")

    slice_json = _first_json_object_slice(coerced)
    if slice_json:
        logger.warning(
            "Fell back to _first_json_object_slice to extract JSON from model output."
        )
        try:
            return EntityExtractionOutput.model_validate_json(slice_json)
        except Exception:
            return EntityExtractionOutput.model_validate(json.loads(slice_json))

    raise ValueError(
        "Could not parse entity extraction JSON from model output. "
        f"First 500 chars: {raw[:500]!r}"
    )


async def run_entity_extraction(
    title: str, maintext: str, *, langfuse_tracing: bool = False
) -> EntityExtractionOutput:
    """Run the entity extraction agent on one article and return structured output.

    Parameters
    ----------
    title : str
        Article headline.
    maintext : str
        Full article body.
    langfuse_tracing : bool, default False
        Whether to enable Langfuse tracing via OpenTelemetry. When ``True``,
        ``init_tracing`` is called before the ADK runner is created, and traces
        are flushed after the run completes.
    """
    if langfuse_tracing:
        from aieng.agent_evals.langfuse import init_tracing  # noqa: PLC0415

        init_tracing(service_name="EntityExtractionAgent")

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
        if langfuse_tracing:
            from aieng.agent_evals.evaluation.trace import flush_traces  # noqa: PLC0415

            flush_traces()
