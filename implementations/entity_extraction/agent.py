"""ADK discovery entrypoint for the Entity Extraction agent.

Exposes a module-level ``root_agent`` so ``adk web`` can discover it.

Examples
--------
Run with ``adk web``:
    uv run adk web --port 8000 --reload --reload_agents implementations/
"""

import logging

from aieng.agent_evals.entity_extraction.agent import create_entity_extraction_agent


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ADK discovery expects a module-level `root_agent`
root_agent = create_entity_extraction_agent()
