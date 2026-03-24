"""Entity extraction agent using Google ADK.

This package provides an LLM-based entity extraction agent that identifies
named entities and mentioned companies from article text.

Example
-------
>>> from aieng.agent_evals.entity_extraction import (
...     create_entity_extraction_agent,
...     EntityExtractionOutput,
... )
>>> agent = create_entity_extraction_agent()
>>> agent.name
'EntityExtractionAgent'
"""

from .agent import create_entity_extraction_agent
from .data import EntityExtractionOutput, NamedEntity

__all__ = [
    "create_entity_extraction_agent",
    "EntityExtractionOutput",
    "NamedEntity",
]
