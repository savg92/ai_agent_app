from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import traceback
from contextlib import asynccontextmanager
from langchain.chains import RetrievalQA
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document as LangchainDocument

# Import utility functions and types
from utils import (
    get_embedding_function,
    get_llm,
    create_qa_chain,
    answer_question,
    create_or_load_vector_db,
    CHROMA_DB_DIRECTORY
)


# --- Global Variables / State (Typed) ---
# These will be initialized via the lifespan manager
vector_db: Optional[VectorStore] = None
qa_chain: Optional[RetrievalQA] = None

# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic: Load the model and initialize the QA chain
    global vector_db, qa_chain
    logging.info("Application startup: Initializing QA chain...")
    try:
        # Initialize using default providers from environment for startup
        embedding_func = get_embedding_function()
        llm = get_llm()
        vector_db = create_or_load_vector_db(embedding_function=embedding_func, force_reload=False)
        if vector_db is None:
            logging.error("Startup failed: Vector database could not be loaded. Run load_db.py first.")
            raise RuntimeError("Vector database could not be loaded.")
        else:
            qa_chain = create_qa_chain(vector_db, llm)
            logging.info("QA Chain initialized successfully.")

    except ValueError as e:
        logging.error(f"Startup failed due to Configuration Error: {e}")
    except Exception as e:
        logging.error(f"Startup failed due to Initialization Error: {e}")
        raise RuntimeError("Application startup failed due to Initialization Error.")

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

class SourceDocument(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

class AskResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]


# --- Dependency for QA Chain ---
# This dependency now just checks if the chain is ready
async def get_initialized_qa_chain() -> RetrievalQA:
    if qa_chain is None:
        logging.error("QA chain accessed before initialization or initialization failed.")
        raise HTTPException(status_code=503, detail="Service Unavailable: QA Chain not ready.")
    # Note: This simple global approach isn't ideal for dynamic provider switching per request
    # without re-initializing, which can be slow. Consider caching chains per provider if needed.
    return qa_chain

# --- API Endpoints ---
@app.post("/ask", response_model=AskResponse)
async def ask_question_endpoint(
    request: AskRequest,
    current_qa_chain: RetrievalQA = Depends(get_initialized_qa_chain) # Inject the initialized chain
    ) -> AskResponse:
    """
    Receives a question, retrieves relevant context, and generates an answer using an LLM.
    """
    logging.info(f"Received question: {request.question}")
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # The endpoint now always uses the globally initialized current_qa_chain

        answer_str: str
        source_docs: List[LangchainDocument]
        answer_str, source_docs = answer_question(current_qa_chain, request.question)

        # Convert source documents to Pydantic model
        response_sources: List[SourceDocument] = [
            SourceDocument(page_content=doc.page_content, metadata=doc.metadata) for doc in source_docs
        ]

        logging.info(f"Generated answer: {answer_str[:100]}...") # Log snippet
        return AskResponse(answer=answer_str, sources=response_sources)

    except ValueError as e:
        logging.error(f"Value Error during question answering: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Error during question answering: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {"status": "ok", "qa_chain_initialized": qa_chain is not None}