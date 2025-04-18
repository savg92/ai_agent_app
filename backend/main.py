import logging
import traceback
import os # Import os
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, Tuple # Ensure Tuple is imported

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain # Ensure this is imported
from langchain_core.documents import Document as LangchainDocument # Add this import
from langchain_core.vectorstores import VectorStore # Add this import
from pydantic import BaseModel # Add this import
from fastapi import FastAPI, HTTPException, Depends # Add these imports
from fastapi.middleware.cors import CORSMiddleware # Add this import

# Import utility functions and types
from utils import (
    get_embedding_function,
    get_llm,
    create_qa_chain,
    answer_question,
    create_or_load_vector_db,
    load_documents_from_directory, # Import document loading function
    DATA_PATH, # Import data path
    CHROMA_DB_DIRECTORY
)


# --- Global Variables / State (Typed) ---
vector_db: Optional[VectorStore] = None
qa_chain: Optional[ConversationalRetrievalChain] = None
# In-memory storage for chat histories {session_id: chat_history}
# WARNING: This is volatile and not suitable for production (lost on restart, not scalable).
# Consider using Redis, a database, or another persistent store for production.
chat_histories: Dict[str, List[Tuple[str, str]]] = {}


# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    # Startup logic: Load the model and initialize the QA chain
    global vector_db, qa_chain
    logging.info("Application startup: Initializing QA chain...")
    try:
        # Initialize using default providers from environment for startup
        embedding_func = get_embedding_function()
        llm = get_llm()

        logging.info(f"Attempting to load vector database from: {CHROMA_DB_DIRECTORY}")
        # Try to load existing DB first
        vector_db = create_or_load_vector_db(embedding_function=embedding_func, force_reload=False)

        # If DB doesn't exist, create it
        if vector_db is None:
            logging.warning(f"Vector database not found at {CHROMA_DB_DIRECTORY}. Attempting to create it...")
            # Ensure the data path exists before trying to load
            if not os.path.exists(DATA_PATH):
                 logging.error(f"Data directory not found at {DATA_PATH}. Cannot create vector database.")
                 raise RuntimeError(f"Data directory not found at {DATA_PATH}. Cannot create vector database.")

            logging.info(f"Loading documents from: {DATA_PATH}")
            documents: List[LangchainDocument]
            failed_files: List[str]
            documents, failed_files = load_documents_from_directory(DATA_PATH)

            if failed_files:
                logging.warning(f"The following files failed to load during DB creation: {failed_files}")

            if not documents:
                logging.error("No documents loaded successfully. Cannot create vector database.")
                raise RuntimeError("No documents loaded successfully. Cannot create vector database.")

            logging.info(f"Creating new vector database with {len(documents)} documents...")
            # Call create_or_load_vector_db again, this time with documents to trigger creation
            vector_db = create_or_load_vector_db(
                documents=documents,
                embedding_function=embedding_func,
                force_reload=False # False is correct, it will create if not existing
            )

            if vector_db is None:
                # This might happen if splitting documents resulted in nothing, or another error occurred
                logging.error("Failed to create vector database even after loading documents.")
                raise RuntimeError("Failed to create vector database.")
            else:
                 logging.info("Vector database created successfully.")

        # If we have a vector_db (either loaded or created), create the QA chain
        qa_chain = create_qa_chain(vector_db, llm)
        logging.info("QA Chain initialized successfully.")

    except ValueError as e:
        # Configuration errors (e.g., missing API keys, invalid model names)
        logging.error(f"Startup failed due to Configuration Error: {e}", exc_info=True)
        # Re-raise or handle specific config errors if needed
        raise RuntimeError(f"Application startup failed due to Configuration Error: {e}")
    except Exception as e:
        # Catch other potential errors during initialization (network, file access, etc.)
        logging.error(f"Startup failed due to Initialization Error: {e}", exc_info=True)
        # It's often useful to re-raise the original exception or a more specific one
        # For simplicity here, we wrap it in a RuntimeError
        raise RuntimeError(f"Application startup failed due to Initialization Error: {e}")

    yield

    # Shutdown logic (if any)
    logging.info("Application shutdown.")
    # Clean up resources here if needed, e.g.:
    # vector_db = None
    # qa_chain = None


# --- Application Setup ---
app = FastAPI(
    title="AI Agent Developer Test API",
    description="API for querying documents using RAG.",
    version="1.0.0",
    lifespan=lifespan
)

# --- CORS Middleware ---
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    # Add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class AskRequest(BaseModel):
    question: str
    session_id: str # Require a session_id from the client


class SourceDocument(BaseModel):
    page_content: str
    metadata: Dict[str, Any]


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    session_id: str # Return session_id to confirm


# --- Dependency for QA Chain ---
async def get_initialized_qa_chain() -> ConversationalRetrievalChain: # Update return type hint
    if qa_chain is None:
        logging.error("QA chain accessed before initialization or initialization failed.")
        raise HTTPException(status_code=503, detail="Service Unavailable: QA Chain not ready.")
    return qa_chain

# --- API Endpoints ---
@app.post("/ask", response_model=AskResponse)
async def ask_question_endpoint(
    request: AskRequest,
    current_qa_chain: ConversationalRetrievalChain = Depends(get_initialized_qa_chain)
) -> AskResponse:
    """
    Receives a question and session_id, retrieves relevant context and chat history,
    generates an answer using an LLM with conversational memory, and updates the history.

    Args:
        request (AskRequest): The request body containing:
            - question (str): The user's question.
            - session_id (str): A unique identifier for the conversation session.
              The backend uses this ID to retrieve and store the chat history.
              The client must generate and manage this ID.
        current_qa_chain (ConversationalRetrievalChain): The initialized QA chain (injected dependency).

    Returns:
        AskResponse: The response containing the generated answer, source documents,
                     and the session_id used.

    Raises:
        HTTPException: If the question or session_id is empty (400),
                       if the QA chain is not ready (503),
                       or if an internal error occurs (500).
    """
    logging.info(f"Received question: {request.question} for session: {request.session_id}")
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not request.session_id:
         raise HTTPException(status_code=400, detail="Session ID cannot be empty.")

    # Retrieve chat history for the session, default to empty list if new session
    # Type hint for clarity
    chat_history: List[Tuple[str, str]] = chat_histories.get(request.session_id, [])
    logging.info(f"Retrieved chat history length for session {request.session_id}: {len(chat_history)}")

    try:
        answer_str: str
        source_docs: List[LangchainDocument]
        # Pass retrieved chat_history to answer_question
        answer_str, source_docs = answer_question(current_qa_chain, request.question, chat_history)

        # Update the chat history for this session
        # Use setdefault to ensure the key exists before appending
        chat_histories.setdefault(request.session_id, []).append((request.question, answer_str))
        logging.info(f"Updated chat history length for session {request.session_id}: {len(chat_histories[request.session_id])}")


        # Convert source documents to Pydantic model
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
    """Basic health check endpoint."""
    return {"status": "ok", "qa_chain_initialized": qa_chain is not None}