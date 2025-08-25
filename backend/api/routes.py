"""API routes for the AI Agent application."""

from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException

from core.models import (
    AskRequest,
    AskResponse,
    RebuildStoreRequest,
    LLMUpdateRequest,
    LLMConfigResponse,
    EmbeddingUpdateRequest,
    EmbeddingConfigResponse,
)
from services import APIHandlerService, VectorStoreManager


router = APIRouter()

# Global state - will be initialized by the main app
api_handler: Optional[APIHandlerService] = None
vector_store_manager: Optional[VectorStoreManager] = None


def get_api_handler() -> APIHandlerService:
    """Dependency to get the API handler service."""
    if api_handler is None:
        raise HTTPException(status_code=503, detail="Service Unavailable: API Handler not ready.")
    return api_handler


@router.post("/ask", response_model=AskResponse)
async def ask_question_endpoint(
    request: AskRequest,
    handler: APIHandlerService = Depends(get_api_handler)
) -> AskResponse:
    """Ask a question and get an answer."""
    return await handler.ask_question(request)


@router.get("/health")
def health_check(handler: APIHandlerService = Depends(get_api_handler)) -> Dict[str, Any]:
    """Health check endpoint."""
    return handler.health_check()


@router.get("/llm", response_model=LLMConfigResponse)
def get_current_llm(handler: APIHandlerService = Depends(get_api_handler)) -> LLMConfigResponse:
    """Get current LLM configuration snapshot (sanitized)."""
    return handler.get_current_llm()


@router.post("/llm", response_model=LLMConfigResponse)
def update_llm(
    request: LLMUpdateRequest,
    handler: APIHandlerService = Depends(get_api_handler)
) -> LLMConfigResponse:
    """Update/switch the LLM provider at runtime and refresh the QA chain."""
    return handler.update_llm(request)


@router.get("/embeddings", response_model=EmbeddingConfigResponse)
def get_current_embeddings(handler: APIHandlerService = Depends(get_api_handler)) -> EmbeddingConfigResponse:
    """Get current embedding configuration snapshot."""
    return handler.get_current_embeddings()


@router.post("/embeddings", response_model=EmbeddingConfigResponse)
def update_embeddings(
    request: EmbeddingUpdateRequest,
    handler: APIHandlerService = Depends(get_api_handler)
) -> EmbeddingConfigResponse:
    """Update/switch the embedding provider at runtime, load or build the corresponding vector store, and refresh the QA chain."""
    return handler.update_embeddings(request)


@router.get("/vector-stores")
def list_vector_stores(handler: APIHandlerService = Depends(get_api_handler)) -> Dict[str, Any]:
    """List all available vector stores."""
    return handler.list_vector_stores()


@router.delete("/vector-stores/{identifier}")
def delete_vector_store(
    identifier: str,
    handler: APIHandlerService = Depends(get_api_handler)
) -> Dict[str, Any]:
    """Delete a specific vector store by identifier."""
    return handler.delete_vector_store(identifier)


@router.delete("/vector-stores")
def delete_all_vector_stores(handler: APIHandlerService = Depends(get_api_handler)) -> Dict[str, Any]:
    """Delete all vector stores."""
    return handler.delete_all_vector_stores()


@router.post("/vector-stores/rebuild")
def rebuild_current_vector_store(
    request: RebuildStoreRequest,
    handler: APIHandlerService = Depends(get_api_handler)
) -> Dict[str, Any]:
    """Rebuild the current vector store with fresh documents."""
    return handler.rebuild_current_vector_store(request)
