"""API routes for the AI Agent application."""

import os
import logging
import traceback
from typing import List, Tuple, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document as LangchainDocument

from core.models import AskRequest, AskResponse, SourceDocument, RebuildStoreRequest
from services import DocumentService, VectorStoreService, QAService, VectorStoreManager
from providers import EmbeddingProviderFactory, LLMProviderFactory
from config import get_paths, get_settings


router = APIRouter()

# Global state - will be initialized by the main app
qa_chain: ConversationalRetrievalChain = None
chat_histories: Dict[str, List[Tuple[str, str]]] = {}
vector_store_manager: VectorStoreManager = None


def get_initialized_qa_chain() -> ConversationalRetrievalChain:
    """Dependency to get the initialized QA chain."""
    if qa_chain is None:
        logging.error("QA chain accessed before initialization or initialization failed.")
        raise HTTPException(status_code=503, detail="Service Unavailable: QA Chain not ready.")
    return qa_chain


@router.post("/ask", response_model=AskResponse)
async def ask_question_endpoint(
    request: AskRequest,
    current_qa_chain: ConversationalRetrievalChain = Depends(get_initialized_qa_chain)
) -> AskResponse:
    """Ask a question and get an answer."""
    logging.info(f"Received question: {request.question} for session: {request.session_id}")
    
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not request.session_id:
        raise HTTPException(status_code=400, detail="Session ID cannot be empty.")

    chat_history: List[Tuple[str, str]] = chat_histories.get(request.session_id, [])
    logging.info(f"Retrieved chat history length for session {request.session_id}: {len(chat_history)}")

    try:
        qa_service = QAService()
        answer_str, source_docs = qa_service.answer_question(
            current_qa_chain, request.question, chat_history
        )

        chat_histories.setdefault(request.session_id, []).append((request.question, answer_str))
        logging.info(f"Updated chat history length for session {request.session_id}: {len(chat_histories[request.session_id])}")

        response_sources: List[SourceDocument] = [
            SourceDocument(page_content=doc.page_content, metadata=doc.metadata) 
            for doc in source_docs
        ]

        logging.info(f"Generated answer for session {request.session_id}: {answer_str[:100]}...")
        return AskResponse(
            answer=answer_str, 
            sources=response_sources, 
            session_id=request.session_id
        )

    except ValueError as e:
        logging.error(f"Value Error during question answering for session {request.session_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Error during question answering for session {request.session_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")


@router.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {"status": "ok", "qa_chain_initialized": qa_chain is not None}


@router.get("/vector-stores")
def list_vector_stores() -> Dict[str, Any]:
    """List all available vector stores."""
    if vector_store_manager is None:
        raise HTTPException(status_code=503, detail="Vector store manager not initialized.")
    
    stores = vector_store_manager.list_available_stores()
    active_config = vector_store_manager.get_active_store_config()
    
    return {
        "stores": stores,
        "active_config": active_config,
        "total_stores": len(stores)
    }


@router.delete("/vector-stores/{identifier}")
def delete_vector_store(identifier: str) -> Dict[str, Any]:
    """Delete a specific vector store by identifier."""
    if vector_store_manager is None:
        raise HTTPException(status_code=503, detail="Vector store manager not initialized.")
    
    success = vector_store_manager.delete_store(identifier)
    if success:
        return {"message": f"Vector store '{identifier}' deleted successfully."}
    else:
        raise HTTPException(status_code=404, detail=f"Vector store '{identifier}' not found.")


@router.delete("/vector-stores")
def delete_all_vector_stores() -> Dict[str, Any]:
    """Delete all vector stores."""
    if vector_store_manager is None:
        raise HTTPException(status_code=503, detail="Vector store manager not initialized.")
    
    vector_store_manager.delete_all_stores()
    return {"message": "All vector stores deleted successfully."}


@router.post("/vector-stores/rebuild")
def rebuild_current_vector_store(request: RebuildStoreRequest) -> Dict[str, Any]:
    """Rebuild the current vector store with fresh documents."""
    global qa_chain
    
    if vector_store_manager is None:
        raise HTTPException(status_code=503, detail="Vector store manager not initialized.")
    
    try:
        paths = get_paths()
        settings = get_settings()
        
        # Load documents
        if not os.path.exists(paths.data_path):
            raise HTTPException(status_code=400, detail=f"Data directory not found at {paths.data_path}.")
        
        document_service = DocumentService()
        documents, failed_files = document_service.load_documents_from_directory(paths.data_path)
        
        if not documents:
            raise HTTPException(status_code=400, detail="No documents loaded successfully.")
        
        # Get current config and services
        embedding_function = EmbeddingProviderFactory.create_embedding_function()
        llm = LLMProviderFactory.create_llm()
        
        # Get current embedding config
        current_config = {
            "provider": settings.embedding_provider,
            "model": getattr(settings, f"{settings.embedding_provider}_embedding_model", None) or 
                    getattr(settings, f"{settings.embedding_provider}_model", None)
        }
        
        # Delete existing store for this config if force is True
        if request.force:
            identifier = vector_store_manager.get_vector_store_identifier(current_config)
            vector_store_manager.delete_store(identifier)
        
        # Create new store
        vector_db = vector_store_manager.get_or_create_store(
            config=current_config,
            documents=documents,
            embedding_function=embedding_function
        )
        
        # Update QA chain
        qa_service = QAService()
        qa_chain = qa_service.create_qa_chain(vector_db, llm)
        vector_store_manager.set_active_store(current_config)
        
        return {
            "message": "Vector store rebuilt successfully.",
            "document_count": len(documents),
            "failed_files": failed_files,
            "config": current_config
        }
        
    except Exception as e:
        logging.error(f"Error rebuilding vector store: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to rebuild vector store: {e}")
