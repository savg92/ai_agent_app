"""Core types and type definitions."""

from typing import Dict, Any, TypedDict, Optional


class EmbeddingConfig(TypedDict):
    """Type definition for embedding configuration."""
    provider: str
    model: Optional[str]
    deployment: Optional[str]
    model_id: Optional[str]
    base_url: Optional[str]
    endpoint: Optional[str]
    region: Optional[str]
