"""Core data models for the AI Agent application."""

from typing import Dict, Any, List, Optional
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


class LLMUpdateRequest(BaseModel):
    """Request model to update/switch LLM provider and settings at runtime."""
    provider: str
    # Generic optional fields used by providers; ignored when not applicable
    llm_temperature: Optional[float] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    # Provider-specific optional fields
    ollama_llm_model: Optional[str] = None
    ollama_base_url: Optional[str] = None
    azure_api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_api_version: Optional[str] = None
    azure_llm_deployment: Optional[str] = None
    bedrock_model_id: Optional[str] = None
    aws_region: Optional[str] = None
    aws_profile: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    lm_studio_model: Optional[str] = None
    lm_studio_base_url: Optional[str] = None
    lm_studio_api_key: Optional[str] = None


class LLMConfigResponse(BaseModel):
    """Response model with current LLM configuration snapshot."""
    provider: str
    details: Dict[str, Any]
