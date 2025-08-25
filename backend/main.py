"""Main application entry point - modular version."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import ConversationalRetrievalChain
from langchain_core.vectorstores import VectorStore

# Import modular components
from config import get_settings, get_paths
from core.types import EmbeddingConfig
from services import DocumentService, VectorStoreService, QAService, VectorStoreManager, APIHandlerService
from services.configuration_service import ConfigurationService
from providers import EmbeddingProviderFactory, LLMProviderFactory
from api.routes import router
import api.routes as routes_module  # Import to access global variables


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    settings = get_settings()
    paths = get_paths()
    
    logging.info("Application startup: Initializing QA chain...")
    
    try:
        # Initialize services
        document_service = DocumentService()
        vector_store_service = VectorStoreService(document_service)
        vector_store_manager = VectorStoreManager(vector_store_service)
        qa_service = QAService()
        config_service = ConfigurationService()
        api_handler_service = APIHandlerService()
        
        # Set global services for routes
        routes_module.vector_store_manager = vector_store_manager
        routes_module.api_handler = api_handler_service
        api_handler_service.set_vector_store_manager(vector_store_manager)
        
        # Get providers
        embedding_func = EmbeddingProviderFactory.create_embedding_function()
        llm = LLMProviderFactory.create_llm()
        current_config = config_service.get_current_embedding_config()

        logging.info(f"Current embedding config: {current_config}")
        
        # Try to get or create vector store
        vector_db: Optional[VectorStore] = None
        
        try:
            vector_db = vector_store_manager.get_or_create_store(
                config=current_config,
                embedding_function=embedding_func
            )
            logging.info("Successfully loaded existing vector store.")
        except ValueError:
            # No existing store or need documents to create new one
            logging.info("No existing vector store found or documents needed. Loading documents...")
            
            if not os.path.exists(paths.data_path):
                logging.error(f"Data directory not found at {paths.data_path}. Cannot create vector database.")
                raise RuntimeError(f"Data directory not found at {paths.data_path}. Cannot create vector database.")
            
            logging.info(f"Loading documents from: {paths.data_path}")
            documents, failed_files = document_service.load_documents_from_directory(paths.data_path)

            if failed_files:
                logging.warning(f"The following files failed to load: {failed_files}")

            if not documents:
                logging.error("No documents loaded successfully. Cannot create vector database.")
                raise RuntimeError("No documents loaded successfully. Cannot create vector database.")

            logging.info(f"Creating new vector store with {len(documents)} documents...")
            vector_db = vector_store_manager.get_or_create_store(
                config=current_config,
                documents=documents,
                embedding_function=embedding_func
            )
            
        # Set this as the active store
        vector_store_manager.set_active_store(current_config)

        # Create QA chain and set it in the handler service
        qa_chain = qa_service.create_qa_chain(vector_db, llm)
        api_handler_service.set_qa_chain(qa_chain)
        
        logging.info("QA Chain initialized successfully.")

    except ValueError as e:
        logging.error(f"Startup failed due to Configuration Error: {e}", exc_info=True)
        raise RuntimeError(f"Application startup failed due to Configuration Error: {e}")
    except Exception as e:
        logging.error(f"Startup failed due to Initialization Error: {e}", exc_info=True)
        raise RuntimeError(f"Application startup failed due to Initialization Error: {e}")

    yield

    logging.info("Application shutdown.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.app_version,
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(router)
    
    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
