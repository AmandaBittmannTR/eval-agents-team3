"""Entity extraction agent.

This module defines the factory used to build the entity extraction agent.

The returned agent is a Google ADK ``LlmAgent`` configured to:

- Accept article text (title + maintext) as input.
- Extract named entities and mentioned companies.
- Return structured output conforming to ``EntityExtractionOutput``.

Examples
--------
>>> from aieng.agent_evals.entity_extraction.agent import create_entity_extraction_agent
>>> agent = create_entity_extraction_agent()
>>> agent.name
'EntityExtractionAgent'
"""

from aieng.agent_evals.configs import Configs
from aieng.agent_evals.entity_extraction.data import EntityExtractionOutput
from google.adk.agents import LlmAgent
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
  in the text or whose ticker symbol appears directly in the text.
- Only include ticker symbols that are **stated in or directly inferable from \
  the text**. Do not look up or guess ticker symbols that are not present.
- Return an empty list if no ticker symbols are mentioned.

### named_entities
For every named entity found in the title or maintext, produce an object with:

| Field          | Description |
|----------------|-------------|
| `entity_group` | One of `ORG` (organisation), `PER` (person), `LOC` (location), `MISC` (miscellaneous). |
| `word`         | The entity text **exactly as it appears** in the source. |
| `normalized`   | A normalized or canonical form if one is available from the text (e.g. a ticker symbol for a company). Set to `null` when no canonical form is present in the text. |

### Guidelines
- Extract **all** entities, not just the most prominent ones.
- Preserve the original surface form in `word`; do not alter capitalisation or spelling.
- Classify courts, government bodies, and regulatory agencies as `ORG`.
- Classify countries, cities, continents, and regions as `LOC`.
- When the same entity appears multiple times, include it **only once**.
- Do not fabricate entities that are not in the text.

## Output
Return a single JSON object matching the configured output schema exactly.
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
        Configured entity extraction agent with ``EntityExtractionOutput``
        as the enforced response schema.
    """
    config = Configs()  # type: ignore[call-arg]
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
        tools=[],
        generate_content_config=GenerateContentConfig(
            temperature=temperature,
            thinking_config=thinking_config,
        ),
        output_schema=EntityExtractionOutput,
    )
