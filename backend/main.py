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
    DATA_PATH,
    CHROMA_DB_DIRECTORY
)


vector_db: Optional[VectorStore] = None
qa_chain: Optional[ConversationalRetrievalChain] = None
chat_histories: Dict[str, List[Tuple[str, str]]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    global vector_db, qa_chain
    logging.info("Application startup: Initializing QA chain...")
    try:
        embedding_func = get_embedding_function()
        llm = get_llm()

        logging.info(f"Attempting to load vector database from: {CHROMA_DB_DIRECTORY}")
        
        # Check if embedding config has changed and we need to rebuild
        from utils import has_embedding_config_changed
        config_changed = has_embedding_config_changed()
        
        if config_changed and os.path.exists(CHROMA_DB_DIRECTORY):
            logging.warning("Embedding configuration has changed. Loading documents to rebuild vector database...")
            if not os.path.exists(DATA_PATH):
                logging.error(f"Data directory not found at {DATA_PATH}. Cannot rebuild vector database.")
                raise RuntimeError(f"Data directory not found at {DATA_PATH}. Cannot rebuild vector database.")
            
            logging.info(f"Loading documents from: {DATA_PATH}")
            documents: List[LangchainDocument]
            failed_files: List[str]
            documents, failed_files = load_documents_from_directory(DATA_PATH)

            if failed_files:
                logging.warning(f"The following files failed to load during DB rebuild: {failed_files}")

            if not documents:
                logging.error("No documents loaded successfully. Cannot rebuild vector database.")
                raise RuntimeError("No documents loaded successfully. Cannot rebuild vector database.")

            logging.info(f"Rebuilding vector database with {len(documents)} documents due to config change...")
            vector_db = create_or_load_vector_db(
                documents=documents,
                embedding_function=embedding_func,
                force_reload=True  # Force rebuild due to config change
            )
        else:
            # Try to load existing vector database
            vector_db = create_or_load_vector_db(embedding_function=embedding_func, force_reload=False)

        if vector_db is None:
            logging.warning(f"Vector database not found at {CHROMA_DB_DIRECTORY}. Attempting to create it...")
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
            vector_db = create_or_load_vector_db(
                documents=documents,
                embedding_function=embedding_func,
                force_reload=False
            )

            if vector_db is None:
                logging.error("Failed to create vector database even after loading documents.")
                raise RuntimeError("Failed to create vector database.")
            else:
                 logging.info("Vector database created successfully.")

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