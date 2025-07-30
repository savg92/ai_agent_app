import logging
import traceback
import os
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, Tuple

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document as LangchainDocument
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from utils import (
    get_embedding_function,
    get_llm,
    create_qa_chain,
    answer_question,
    create_or_load_vector_db,
    load_documents_from_directory,
    get_current_embedding_config,
    VectorStoreManager,
    DATA_PATH,
    CHROMA_DB_DIRECTORY
)


vector_db: Optional[VectorStore] = None
qa_chain: Optional[ConversationalRetrievalChain] = None
chat_histories: Dict[str, List[Tuple[str, str]]] = {}
vector_store_manager: Optional[VectorStoreManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    global vector_db, qa_chain, vector_store_manager
    logging.info("Application startup: Initializing QA chain...")
    try:
        # Initialize vector store manager
        vector_store_manager = VectorStoreManager()
        
        embedding_func = get_embedding_function()
        llm = get_llm()
        current_config = get_current_embedding_config()

        logging.info(f"Current embedding config: {current_config}")
        
        # Try to get or create vector store using the new manager
        try:
            vector_db = vector_store_manager.get_or_create_store(
                config=current_config,
                embedding_function=embedding_func
            )
            logging.info("Successfully loaded existing vector store.")
        except ValueError as e:
            # No existing store or need documents to create new one
            logging.info("No existing vector store found or documents needed. Loading documents...")
            if not os.path.exists(DATA_PATH):
                logging.error(f"Data directory not found at {DATA_PATH}. Cannot create vector database.")
                raise RuntimeError(f"Data directory not found at {DATA_PATH}. Cannot create vector database.")
            
            logging.info(f"Loading documents from: {DATA_PATH}")
            documents: List[LangchainDocument]
            failed_files: List[str]
            documents, failed_files = load_documents_from_directory(DATA_PATH)

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

        qa_chain = create_qa_chain(vector_db, llm)
        logging.info("QA Chain initialized successfully.")

    except ValueError as e:
        logging.error(f"Startup failed due to Configuration Error: {e}", exc_info=True)
        raise RuntimeError(f"Application startup failed due to Configuration Error: {e}")
    except Exception as e:
        logging.error(f"Startup failed due to Initialization Error: {e}", exc_info=True)
        raise RuntimeError(f"Application startup failed due to Initialization Error: {e}")

    yield

    logging.info("Application shutdown.")


app = FastAPI(
    title="AI Agent Developer Test API",
    description="API for querying documents using RAG.",
    version="1.0.0",
    lifespan=lifespan
)

origins = [
    "http://localhost:3000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    session_id: str


class SourceDocument(BaseModel):
    page_content: str
    metadata: Dict[str, Any]


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    session_id: str


async def get_initialized_qa_chain() -> ConversationalRetrievalChain:
    if qa_chain is None:
        logging.error("QA chain accessed before initialization or initialization failed.")
        raise HTTPException(status_code=503, detail="Service Unavailable: QA Chain not ready.")
    return qa_chain

@app.post("/ask", response_model=AskResponse)
async def ask_question_endpoint(
    request: AskRequest,
    current_qa_chain: ConversationalRetrievalChain = Depends(get_initialized_qa_chain)
) -> AskResponse:
    logging.info(f"Received question: {request.question} for session: {request.session_id}")
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not request.session_id:
         raise HTTPException(status_code=400, detail="Session ID cannot be empty.")

    chat_history: List[Tuple[str, str]] = chat_histories.get(request.session_id, [])
    logging.info(f"Retrieved chat history length for session {request.session_id}: {len(chat_history)}")

    try:
        answer_str: str
        source_docs: List[LangchainDocument]
        answer_str, source_docs = answer_question(current_qa_chain, request.question, chat_history)

        chat_histories.setdefault(request.session_id, []).append((request.question, answer_str))
        logging.info(f"Updated chat history length for session {request.session_id}: {len(chat_histories[request.session_id])}")

        response_sources: List[SourceDocument] = [
            SourceDocument(page_content=doc.page_content, metadata=doc.metadata) for doc in source_docs
        ]

        logging.info(f"Generated answer for session {request.session_id}: {answer_str[:100]}...")
        return AskResponse(answer=answer_str, sources=response_sources, session_id=request.session_id)

    except ValueError as e:
        logging.error(f"Value Error during question answering for session {request.session_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Error during question answering for session {request.session_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")


@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {"status": "ok", "qa_chain_initialized": qa_chain is not None}


@app.get("/vector-stores")
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


@app.delete("/vector-stores/{identifier}")
def delete_vector_store(identifier: str) -> Dict[str, Any]:
    """Delete a specific vector store by identifier."""
    if vector_store_manager is None:
        raise HTTPException(status_code=503, detail="Vector store manager not initialized.")
    
    success = vector_store_manager.delete_store(identifier)
    if success:
        return {"message": f"Vector store '{identifier}' deleted successfully."}
    else:
        raise HTTPException(status_code=404, detail=f"Vector store '{identifier}' not found.")


@app.delete("/vector-stores")
def delete_all_vector_stores() -> Dict[str, Any]:
    """Delete all vector stores."""
    if vector_store_manager is None:
        raise HTTPException(status_code=503, detail="Vector store manager not initialized.")
    
    vector_store_manager.delete_all_stores()
    return {"message": "All vector stores deleted successfully."}


class RebuildStoreRequest(BaseModel):
    force: bool = False


@app.post("/vector-stores/rebuild")
def rebuild_current_vector_store(request: RebuildStoreRequest) -> Dict[str, Any]:
    """Rebuild the current vector store with fresh documents."""
    global vector_db, qa_chain
    
    if vector_store_manager is None:
        raise HTTPException(status_code=503, detail="Vector store manager not initialized.")
    
    try:
        # Load documents
        if not os.path.exists(DATA_PATH):
            raise HTTPException(status_code=400, detail=f"Data directory not found at {DATA_PATH}.")
        
        documents, failed_files = load_documents_from_directory(DATA_PATH)
        
        if not documents:
            raise HTTPException(status_code=400, detail="No documents loaded successfully.")
        
        # Get current config and embedding function
        current_config = get_current_embedding_config()
        embedding_func = get_embedding_function()
        llm = get_llm()
        
        # Delete existing store for this config if force is True
        if request.force:
            identifier = vector_store_manager.get_vector_store_identifier(current_config)
            vector_store_manager.delete_store(identifier)
        
        # Create new store
        vector_db = vector_store_manager.get_or_create_store(
            config=current_config,
            documents=documents,
            embedding_function=embedding_func
        )
        
        # Update QA chain
        qa_chain = create_qa_chain(vector_db, llm)
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