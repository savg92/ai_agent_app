# Copilot Instructions

## About this Project

This is a Retrieval-Augmented Generation (RAG) application that uses a FastAPI backend to answer questions about documents. The core logic is built with Langchain.

The architecture consists of:

- A `frontend/` directory (currently a placeholder).
- A `backend/` directory containing the Python FastAPI application.
- A `data/` directory for source documents (.txt, .json, .csv).

The backend (`backend/main.py`) exposes an `/ask` endpoint that takes a question and a session ID. It maintains a conversational history for each session.

The core RAG pipeline is in `backend/utils.py`:

1.  **Document Loading**: `load_documents_from_directory` loads files from `data/`.
2.  **Vector DB**: `create_or_load_vector_db` uses ChromaDB to store document embeddings. The database is stored in `backend/chroma_db_store/`.
3.  **Embeddings & LLMs**: `get_embedding_function` and `get_llm` dynamically select providers based on environment variables. This is the primary way the application is configured.
4.  **QA Chain**: `create_qa_chain` builds the `ConversationalRetrievalChain` from Langchain.

## Getting Started & Development Workflow

1.  **Setup**: Follow the instructions in `README.md` to set up the Python virtual environment and install dependencies from `backend/requirements.txt`.
2.  **Configuration**: Copy `backend/.env.example` to `backend/.env` and configure your desired `EMBEDDING_PROVIDER` and `LLM_PROVIDER`. Add the necessary API keys and model names for your chosen providers.
3.  **Run the backend**:
    ```bash
    cd backend
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```
4.  **Populating the Database**: The vector database in `backend/chroma_db_store/` is created automatically when the server starts if it doesn't exist. To force a rebuild, you can either delete the directory or run the `load_db.py` script:
    ```bash
    python backend/load_db.py --reload
    ```

## Key Components & Patterns

- **`backend/main.py`**: This is the entry point for the FastAPI application. It handles API requests, manages the application lifecycle (loading the model on startup), and maintains in-memory chat histories.
- **`backend/utils.py`**: This is the core of the application.
  - The `get_embedding_function` and `get_llm` functions are the key integration points for adding new providers. To add a new provider, you would add a new `elif provider == "new_provider":` block to these functions and implement the necessary logic, pulling configuration from environment variables.
  - It uses a factory pattern to instantiate the correct LLM and embedding clients.
- **`backend/load_db.py`**: A utility script for manually managing the ChromaDB vector store.
- **`data/`**: This directory holds all the source material for the RAG system. Add any new documents here.

## Configuration via Environment Variables

The application is configured almost entirely through environment variables defined in `backend/.env`. This is the main way to switch between different models and services.

- `EMBEDDING_PROVIDER`: (e.g., `openai`, `ollama`, `azure`, `bedrock`, `lmstudio`)
- `LLM_PROVIDER`: (e.g., `openai`, `ollama`, `azure`, `bedrock`, `lmstudio`)
- Provider-specific variables like `OPENAI_API_KEY`, `OLLAMA_LLM_MODEL`, `AZURE_OPENAI_ENDPOINT`, etc., are required depending on the selected providers.

When adding a new feature that requires configuration, prefer adding new environment variables and loading them in `utils.py` or `main.py`.


<!-- use uv -->
