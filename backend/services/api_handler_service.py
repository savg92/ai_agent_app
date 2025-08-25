"""API handler service for managing all route business logic."""

import os
import logging
import traceback
from typing import List, Tuple, Dict, Any, Optional, cast
import threading
from fastapi import HTTPException
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document as LangchainDocument

from core.models import (
    AskRequest,
    AskResponse,
    SourceDocument,
    RebuildStoreRequest,
    LLMUpdateRequest,
    LLMConfigResponse,
    EmbeddingUpdateRequest,
    EmbeddingConfigResponse,
)
from core.types import EmbeddingConfig
from services import DocumentService, VectorStoreService, QAService, VectorStoreManager
from providers import EmbeddingProviderFactory, LLMProviderFactory
from config import get_paths, get_settings


class APIHandlerService:
    """Service for handling all API route business logic."""
    
    def __init__(self):
        # Global state - will be initialized by the main app
        self.qa_chain: Optional[ConversationalRetrievalChain] = None
        self.chat_histories: Dict[str, List[Tuple[str, str]]] = {}
        self.vector_store_manager: Optional[VectorStoreManager] = None
        self.qa_chain_lock = threading.Lock()
    
    def set_qa_chain(self, qa_chain: ConversationalRetrievalChain) -> None:
        """Set the QA chain."""
        with self.qa_chain_lock:
            self.qa_chain = qa_chain
    
    def set_vector_store_manager(self, vector_store_manager: VectorStoreManager) -> None:
        """Set the vector store manager."""
        self.vector_store_manager = vector_store_manager
    
    def get_initialized_qa_chain(self) -> ConversationalRetrievalChain:
        """Get the initialized QA chain."""
        if self.qa_chain is None:
            logging.error("QA chain accessed before initialization or initialization failed.")
            raise HTTPException(status_code=503, detail="Service Unavailable: QA Chain not ready.")
        return self.qa_chain
    
    async def ask_question(self, request: AskRequest) -> AskResponse:
        """Handle ask question request."""
        logging.info(f"Received question: {request.question} for session: {request.session_id}")
        
        if not request.question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        if not request.session_id:
            raise HTTPException(status_code=400, detail="Session ID cannot be empty.")

        current_qa_chain = self.get_initialized_qa_chain()
        chat_history: List[Tuple[str, str]] = self.chat_histories.get(request.session_id, [])
        logging.info(f"Retrieved chat history length for session {request.session_id}: {len(chat_history)}")

        try:
            qa_service = QAService()
            answer_str, source_docs = qa_service.answer_question(
                current_qa_chain, request.question, chat_history
            )

            self.chat_histories.setdefault(request.session_id, []).append((request.question, answer_str))
            logging.info(f"Updated chat history length for session {request.session_id}: {len(self.chat_histories[request.session_id])}")

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

    def health_check(self) -> Dict[str, Any]:
        """Handle health check request."""
        return {"status": "ok", "qa_chain_initialized": self.qa_chain is not None}

    def get_current_llm(self) -> LLMConfigResponse:
        """Handle get current LLM configuration request."""
        settings = get_settings()
        return LLMConfigResponse(provider=settings.llm_provider, details=settings.current_llm_config())

    def update_llm(self, request: LLMUpdateRequest) -> LLMConfigResponse:
        """Handle update LLM provider request."""
        if self.vector_store_manager is None:
            raise HTTPException(status_code=503, detail="Vector store manager not initialized.")

        # Update settings in-memory
        settings = get_settings()
        settings.update_llm_settings(
            request.provider,
            llm_temperature=getattr(request, 'llm_temperature', None),
            openai_api_key=request.api_key,
            ollama_llm_model=request.ollama_llm_model or request.model,
            ollama_base_url=request.ollama_base_url or request.base_url,
            azure_api_key=request.azure_api_key,
            azure_endpoint=request.azure_endpoint,
            azure_api_version=request.azure_api_version,
            azure_llm_deployment=request.azure_llm_deployment,
            bedrock_model_id=request.bedrock_model_id,
            aws_region=request.aws_region,
            aws_profile=request.aws_profile,
            aws_access_key_id=request.aws_access_key_id,
            aws_secret_access_key=request.aws_secret_access_key,
            lm_studio_model=request.lm_studio_model or request.model,
            lm_studio_base_url=request.lm_studio_base_url or request.base_url,
            lm_studio_api_key=request.lm_studio_api_key or request.api_key,
        )

        try:
            # Create new LLM instance
            llm = LLMProviderFactory.create_llm()
            # Rebuild QA chain using existing active vector store
            active_config = self.vector_store_manager.get_active_store_config()
            if not active_config:
                raise HTTPException(status_code=503, detail="No active vector store configured.")
            embedding_func = EmbeddingProviderFactory.create_embedding_function()
            vector_db = self.vector_store_manager.get_or_create_store(cast(EmbeddingConfig, active_config), embedding_function=embedding_func)
            qa_service = QAService()
            new_chain = qa_service.create_qa_chain(vector_db, llm)
            # Atomic swap under lock
            with self.qa_chain_lock:
                self.qa_chain = new_chain
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logging.error(f"Failed to update LLM: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to update LLM: {e}")

        return LLMConfigResponse(provider=settings.llm_provider, details=settings.current_llm_config())

    def get_current_embeddings(self) -> EmbeddingConfigResponse:
        """Handle get current embedding configuration request."""
        settings = get_settings()
        return EmbeddingConfigResponse(provider=settings.embedding_provider, details=settings.current_embedding_config())

    def update_embeddings(self, request: EmbeddingUpdateRequest) -> EmbeddingConfigResponse:
        """Handle update embedding provider request."""
        if self.vector_store_manager is None:
            raise HTTPException(status_code=503, detail="Vector store manager not initialized.")

        settings = get_settings()
        # Apply in-memory settings update
        settings.update_embedding_settings(
            request.provider,
            openai_api_key=request.api_key,
            ollama_embedding_model=request.ollama_embedding_model or request.model,
            ollama_base_url=request.ollama_base_url or request.base_url,
            azure_api_key=request.azure_api_key,
            azure_endpoint=request.azure_endpoint,
            azure_api_version=request.azure_api_version,
            azure_embedding_deployment=request.azure_embedding_deployment,
            bedrock_embedding_model_id=request.bedrock_embedding_model_id,
            aws_region=request.aws_region,
            aws_profile=request.aws_profile,
            aws_access_key_id=request.aws_access_key_id,
            aws_secret_access_key=request.aws_secret_access_key,
            lm_studio_embedding_model=request.lm_studio_embedding_model or request.model,
            lm_studio_base_url=request.lm_studio_base_url or request.base_url,
            lm_studio_api_key=request.lm_studio_api_key or request.api_key,
        )

        try:
            # Build embedding function for the selected provider
            embedding_func = EmbeddingProviderFactory.create_embedding_function()
            # Derive current embedding config and resolve or create vector store
            current_config = cast(EmbeddingConfig, settings.current_embedding_config())
            # If existing store exists, load it. Otherwise, create from documents.
            try:
                vector_db = self.vector_store_manager.get_or_create_store(
                    config=current_config,
                    embedding_function=embedding_func,
                )
            except ValueError:
                # Need documents to create new store
                paths = get_paths()
                if not os.path.exists(paths.data_path):
                    raise HTTPException(status_code=400, detail=f"Data directory not found at {paths.data_path}.")
                doc_service = DocumentService()
                documents, failed_files = doc_service.load_documents_from_directory(paths.data_path)
                if not documents:
                    raise HTTPException(status_code=400, detail="No documents loaded successfully to build vector store.")
                vector_db = self.vector_store_manager.get_or_create_store(
                    config=current_config,
                    documents=documents,
                    embedding_function=embedding_func,
                )
            # Recreate QA chain using the active LLM
            llm = LLMProviderFactory.create_llm()
            qa_service = QAService()
            new_chain = qa_service.create_qa_chain(vector_db, llm)
            with self.qa_chain_lock:
                self.qa_chain = new_chain
            # Mark as active
            self.vector_store_manager.set_active_store(current_config)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logging.error(f"Failed to update embeddings: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to update embeddings: {e}")

        return EmbeddingConfigResponse(provider=settings.embedding_provider, details=settings.current_embedding_config())

    def list_vector_stores(self) -> Dict[str, Any]:
        """Handle list all available vector stores request."""
        if self.vector_store_manager is None:
            raise HTTPException(status_code=503, detail="Vector store manager not initialized.")
        
        stores = self.vector_store_manager.list_available_stores()
        active_config = self.vector_store_manager.get_active_store_config()
        
        return {
            "stores": stores,
            "active_config": active_config,
            "total_stores": len(stores)
        }

    def delete_vector_store(self, identifier: str) -> Dict[str, Any]:
        """Handle delete a specific vector store request."""
        if self.vector_store_manager is None:
            raise HTTPException(status_code=503, detail="Vector store manager not initialized.")
        
        success = self.vector_store_manager.delete_store(identifier)
        if success:
            return {"message": f"Vector store '{identifier}' deleted successfully."}
        else:
            raise HTTPException(status_code=404, detail=f"Vector store '{identifier}' not found.")

    def delete_all_vector_stores(self) -> Dict[str, Any]:
        """Handle delete all vector stores request."""
        if self.vector_store_manager is None:
            raise HTTPException(status_code=503, detail="Vector store manager not initialized.")
        
        self.vector_store_manager.delete_all_stores()
        return {"message": "All vector stores deleted successfully."}

    def rebuild_current_vector_store(self, request: RebuildStoreRequest) -> Dict[str, Any]:
        """Handle rebuild the current vector store request."""
        if self.vector_store_manager is None:
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
            current_config = cast(EmbeddingConfig, settings.current_embedding_config())
            
            # Delete existing store for this config if force is True
            if request.force:
                identifier = self.vector_store_manager.get_vector_store_identifier(current_config)
                self.vector_store_manager.delete_store(identifier)
            
            # Create new store
            vector_db = self.vector_store_manager.get_or_create_store(
                config=current_config,
                documents=documents,
                embedding_function=embedding_function
            )
            
            # Update QA chain
            qa_service = QAService()
            new_chain = qa_service.create_qa_chain(vector_db, llm)
            with self.qa_chain_lock:
                self.qa_chain = new_chain
            self.vector_store_manager.set_active_store(current_config)
            
            return {
                "message": "Vector store rebuilt successfully.",
                "document_count": len(documents),
                "failed_files": failed_files,
                "config": current_config
            }
            
        except Exception as e:
            logging.error(f"Error rebuilding vector store: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to rebuild vector store: {e}")
