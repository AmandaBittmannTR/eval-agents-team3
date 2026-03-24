"""Data models for the entity extraction agent.

Defines the structured output schema that the agent must conform to.
"""

from enum import Enum

from pydantic import BaseModel, Field


class EntityGroup(str, Enum):
    """Supported named-entity categories."""

    MISC = "MISC"
    ORG = "ORG"
    PER = "PER"
    LOC = "LOC"


class NamedEntity(BaseModel):
    """A single named entity extracted from text.

    Attributes
    ----------
    entity_group : EntityGroup
        The category of the entity (MISC, ORG, PER, LOC).
    word : str
        The entity text as it appears in the source.
    normalized : str | None
        Normalized form of the entity (e.g. a ticker symbol for a company).
        ``None`` when no canonical form is known from the text.
    """

    entity_group: EntityGroup = Field(
        description="Category of the entity: 'ORG' for organisations, 'PER' for people, "
        "'LOC' for locations, 'MISC' for everything else.",
    )
    word: str = Field(description="The entity text exactly as it appears in the source.")
    normalized: str | None = Field(
        default=None,
        description="Normalized or canonical form of the entity if available from the text "
        "(e.g. a ticker symbol for a company). None when not available.",
    )


class EntityExtractionOutput(BaseModel):
    """Structured output returned by the entity extraction agent.

    Attributes
    ----------
    mentioned_companies : list[str]
        Ticker symbols of companies explicitly mentioned in the text.
    named_entities : list[NamedEntity]
        All named entities extracted from the text.
    """

    mentioned_companies: list[str] = Field(
        default_factory=list,
        description="Ticker symbols of companies explicitly mentioned or identifiable from the text.",
    )
    named_entities: list[NamedEntity] = Field(
        default_factory=list,
        description="All named entities extracted from the title and article text.",
    )
