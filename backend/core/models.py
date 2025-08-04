"""Core data models for the AI Agent application."""

from typing import Dict, Any, List
from pydantic import BaseModel


class AskRequest(BaseModel):
    """Request model for asking questions."""
    question: str
    session_id: str


class SourceDocument(BaseModel):
    """Source document model for responses."""
    page_content: str
    metadata: Dict[str, Any]


class AskResponse(BaseModel):
    """Response model for question answers."""
    answer: str
    sources: List[SourceDocument]
    session_id: str


class RebuildStoreRequest(BaseModel):
    """Request model for rebuilding vector stores."""
    force: bool = False
