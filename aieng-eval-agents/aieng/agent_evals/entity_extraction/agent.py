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
import time
import uuid

from aieng.agent_evals.configs import Configs
from aieng.agent_evals.entity_extraction.entity_extraction_models import EntityExtractionOutput
from aieng.agent_evals.token_tracker import TokenTracker, TokenUsage
from aieng.agent_evals.tools import create_google_search_tool
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.genai.types import GenerateContentConfig, ThinkingConfig
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_DEFAULT_AGENT_DESCRIPTION = (
    "Extracts named entities and mentioned companies from article text."
)

EXTRACTION_PROMPT = """\
You are a named-entity recognition (NER) system. Your task is to read the \
provided article and extract named entities, mimicking how a standard NER \
model (like BERT-NER) would tag text.

## Input

You will receive a JSON object with two fields:
- `title`: the article headline.
- `maintext`: the full article body.

## Entity Group Classification Rules

Classify every extracted entity into exactly one of these four groups:

### ORG
- Publicly traded companies: Apple, Google, Amazon, Netflix, Verizon.
- Stock ticker symbols that appear literally in the article: AAPL, GOOG, VZ.
- Do **NOT** include: universities, research labs, government agencies, courts, \
regulatory bodies, academic journals, non-profit organisations, or generic \
industry terms.

### PER
- Named individuals mentioned by name: "Tim Cook", "Trump", "Colin Camerer".
- Include last-name-only references (e.g. "Trump") if the person is clearly \
identified.

### LOC
- Countries: Canada, America, China.
- Cities: New York, San Francisco, Louisville, Miami.
- Other well-known geographic locations: Wall Street, Silicon Valley, Mars.
- Note: U.S. state names and sub-national regions (California, Florida, \
British Columbia) **can be LOC or MISC** depending on context. Use LOC when \
the state is used as a geographic reference; use MISC when it modifies \
something else (e.g. "Florida man", "California law").

### MISC
- Product names: iPhone, Apple Watch Series 2, MacBook Pro.
- Technology and platform names: Bluetooth, Android, iOS.
- Consumer brands: Beats, Parkside, Kickstarter.
- Nationalities and demonyms: Chinese, American, Republican, Democratic.

## Extraction Scope

Be **selective** -- only extract entities that a standard NER tagger would \
confidently tag. Focus on the **most prominent** named entities in the article.

**Extract:**
- Companies and their ticker symbols (as separate entities).
- Named individuals central to the story.
- Geographic locations explicitly named.
- Specific product names, brand names, and well-known technology names.
- Nationalities and demonyms used as proper adjectives.

**Do NOT extract:**
- Generic descriptors, common nouns, adjectives, dates, or numbers.
- Job titles, roles, or occupations on their own (e.g. "CEO", "analyst").
- Vague or generic references ("the company", "the government", "officials").
- Organisations that are not companies (universities, courts, agencies, NGOs).
- Industry jargon or technical terms that are not proper nouns.
- Entities mentioned only in passing or in boilerplate/footer text.

## Word Form Rules

**Use the shortest natural form** of each entity as it appears in the text:
- Write "Apple" not "Apple Inc" (unless "Apple Inc" is the exact surface form used).
- Write "Google" not "Google LLC".
- Write "Verizon" not "Verizon Communications Inc".
- Preserve original capitalisation.
- When the same entity appears multiple times, include it **only once**.

## mentioned_companies

- List ticker symbols for `ORG` entities that are publicly traded.
- Include a ticker if it appears explicitly in the article text, or if you \
resolve it via the ``lookup_ticker`` tool.
- **Only include tickers when that company's ticker appears explicitly in the \
article text** (e.g. ``(AAPL)``, ``GOOG``, ``VZ``). If the article only \
mentions the company name and not a ticker, include the company as an ORG \
entity but **do not** add a ticker to ``mentioned_companies`` unless the \
ticker symbol literally appears in the article.

## Tool-Calling Protocol (lookup_ticker)

1. Scan the article for `ORG` entities (publicly traded companies).
2. For each ORG, call ``lookup_ticker(company_name)`` with **just the company \
name** (e.g. "Apple", "Wells Fargo") -- not a full search query.
3. Use the returned ticker as the ``normalized`` value for that entity.
4. After all lookups, produce your final JSON.

### Do NOT look up
- Universities, research labs, academic journals.
- Cities, countries, geographic regions.
- Courts, regulators, government agencies.
- Individual people.
- Consumer brands that are not publicly traded.

### Critical constraints
- Complete ALL lookups BEFORE producing your final JSON.
- Do NOT output partial JSON between tool calls.

## Output Format

Your final response must be a **single JSON object** (no prose, no markdown \
fences, no commentary):

```
{
  "mentioned_companies": ["AAPL"],
  "named_entities": [
    {"entity_group": "ORG", "word": "Apple", "normalized": "AAPL"},
    {"entity_group": "ORG", "word": "AAPL", "normalized": "AAPL"},
    {"entity_group": "PER", "word": "Tim Cook", "normalized": null},
    {"entity_group": "LOC", "word": "New York", "normalized": null},
    {"entity_group": "MISC", "word": "iPhone", "normalized": null},
    {"entity_group": "MISC", "word": "American", "normalized": null}
  ]
}
```

Field rules:
- ``mentioned_companies``: array of ticker symbol strings. Only include \
tickers that appear literally in the article text.
- ``named_entities``: array of objects, each with:
  - ``entity_group``: one of ``"ORG"``, ``"PER"``, ``"LOC"``, ``"MISC"``.
  - ``word``: shortest natural surface form from the article.
  - ``normalized``: ticker symbol string for ORG entities, or ``null``.
"""


class EntityExtractionResponse(BaseModel):
    """Response from the entity extraction agent.

    Attributes
    ----------
    output : EntityExtractionOutput
        The extracted entities and mentioned companies.
    total_duration_ms : int
        Total execution time in milliseconds.
    token_usage : TokenUsage | None
        Token usage statistics for this extraction call.
    """

    output: EntityExtractionOutput
    total_duration_ms: int = 0
    token_usage: TokenUsage | None = None


def create_entity_extraction_agent(
    name: str = "EntityExtractionAgent",
    *,
    description: str | None = None,
    instructions: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    thinking_budget: int = 2048,
    use_ticker_cache: bool = True,
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
    use_ticker_cache : bool, default True
        When True, use a local ticker cache (with Google Search fallback)
        instead of pure Google Search. New discoveries are appended to
        ``nasdaq_tickers.json``.

    Returns
    -------
    LlmAgent
        Configured entity extraction agent. JSON structure is enforced via
        prompt-embedded schema and example. Post-hoc parsing fallbacks in
        ``run_entity_extraction`` provide defence-in-depth.
    """
    config = Configs()  # type: ignore[call-arg]
    resolved_model = model or config.default_worker_model

    if use_ticker_cache:
        from aieng.agent_evals.entity_extraction.ticker_cache import create_ticker_lookup_tool

        tools = [create_ticker_lookup_tool()]
    else:
        search_tool = create_google_search_tool(config)
        tools = [search_tool]

    thinking_config = None
    if thinking_budget > 0:
        model_lower = resolved_model.lower()
        if "gemini-2.5" in model_lower or "gemini-3" in model_lower:
            thinking_config = ThinkingConfig(thinking_budget=thinking_budget)

    prompt = instructions or EXTRACTION_PROMPT

    return LlmAgent(
        name=name,
        description=description or _DEFAULT_AGENT_DESCRIPTION,
        model=resolved_model,
        instruction=prompt,
        tools=tools,
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
    title: str,
    maintext: str,
    *,
    use_ticker_cache: bool = True,
) -> EntityExtractionResponse:
    """Run the entity extraction agent on one article and return structured output.

    Parameters
    ----------
    title : str
        Article headline.
    maintext : str
        Article body text.
    use_ticker_cache : bool, default True
        When True, use the local ticker cache with Google Search fallback.
        New discoveries are appended to ``nasdaq_tickers.json``.
    """
    config = Configs()  # type: ignore[call-arg]
    agent = create_entity_extraction_agent(
        use_ticker_cache=use_ticker_cache,
    )
    token_tracker = TokenTracker(model=config.default_worker_model)
    session_service = InMemorySessionService()
    runner = Runner(
        app_name=agent.name,
        agent=agent,
        session_service=session_service,
        auto_create_session=True,
    )
    start_time = time.time()
    try:
        payload = json.dumps({"title": title, "maintext": maintext}, ensure_ascii=False)
        message = types.Content(parts=[types.Part(text=payload)], role="user")
        final_text: str | None = None
        async for event in runner.run_async(
            session_id=str(uuid.uuid4()),
            user_id="entity_extraction",
            new_message=message,
        ):
            token_tracker.add_from_event(event)
            if not event.is_final_response():
                continue
            chunk = _final_response_text_from_event(event)
            # ADK may emit multiple "final" events; later ones can be empty and must not
            # overwrite a valid JSON response from an earlier turn.
            if chunk:
                final_text = chunk

        if not final_text or not final_text.strip():
            raise RuntimeError("Entity extraction produced no output.")

        total_duration_ms = int((time.time() - start_time) * 1000)
        return EntityExtractionResponse(
            output=_parse_entity_output(final_text),
            total_duration_ms=total_duration_ms,
            token_usage=token_tracker.usage,
        )
    finally:
        await runner.close()
