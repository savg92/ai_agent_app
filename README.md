# AI Agent RAG Application

This project implements a Retrieval-Augmented Generation (RAG) system using a Python backend built with FastAPI and Langchain. It allows users to ask questions about documents stored locally, leveraging various LLM and embedding providers like Ollama, OpenAI, Azure, Bedrock, and LM Studio.

## Features

- **RAG Pipeline:** Retrieves relevant document chunks from a vector database (ChromaDB) and uses a Language Model (LLM) to generate answers based on the retrieved context.
- **Multi-Vector-Store Architecture:** Each embedding configuration gets its own isolated vector database, eliminating compatibility issues when switching between models.
- **Conversational Memory:** Maintains chat history per session, allowing for follow-up questions.
- **FastAPI Backend:** Provides API endpoints to interact with the RAG system and manage vector stores.
- **Multiple Provider Support:** Configurable to use different providers for embeddings (OpenAI, Ollama, Azure, Bedrock, HuggingFace, LM Studio) and LLMs (OpenAI, Ollama, Azure, Bedrock, LM Studio) via environment variables.
- **Automatic Database Creation:** The backend automatically creates vector databases on first startup and when switching embedding models.
- **Vector Store Management:** CLI and API tools for listing, switching, and managing multiple vector stores.
- **CORS Enabled:** Allows requests from specified frontend origins (e.g., `http://localhost:3000`, `http://localhost:5173`).

## Project Structure

```
.
├── README.md
├── backend/
│   ├── main.py           # FastAPI application, API endpoints, startup logic
│   ├── utils.py          # Core logic for loading data, embeddings, LLMs, QA chain, VectorStoreManager
│   ├── manage_stores.py  # CLI tool for managing vector stores
│   ├── load_db.py        # (Optional) Script to manually load/reload the vector DB
│   ├── rebuild_db.py     # (Optional) Script to rebuild the vector DB
│   ├── requirements.txt  # Python dependencies
│   ├── .env.example      # Example environment variable configuration
│   ├── .env              # Local environment variables (ignored by git)
│   ├── chroma_db_store/  # Legacy vector database (will be migrated)
│   └── vector_stores/    # Multi-store vector databases organized by config
│       ├── openai_text-embedding-ada-002/     # Example: OpenAI store
│       ├── ollama_nomic-embed-text/            # Example: Ollama store
│       └── lmstudio_text-embedding-model/     # Example: LM Studio store
├── data/
│   ├── local_news.json   # Example data file
│   └── tax_policies.txt  # Example data file
└── frontend/             # (Placeholder for frontend application)
```

## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/savg92/ai_agent_app
    cd ai_agent_app
    ```

2.  **Navigate to the backend directory:**

    ```bash
    cd backend
    ```

3.  **Create and activate a virtual environment:**

    - Using `venv`:
      ```bash
      python -m venv .venv
      source .venv/bin/activate
      ```
    - Using `uv` (recommended):
      ```bash
      uv venv .venv
      source .venv/bin/activate
      ```
    - On macOS/Linux:
      ```bash
      source .venv/bin/activate
      ```
    - On Windows:
      ```bash
      .venv\Scripts\activate
      ```
    - _Note:_ If you encounter issues with `uv`, you can use `pip` to create a virtual environment:
      ```bash
      python -m venv .venv
      source .venv/bin/activate
      ```
    - _Note:_ If you are using a different environment manager (like Conda), create and activate your environment accordingly:

      ```bash
      conda create -n myenv python=3.9
      conda activate myenv
      ```

    - Or using your preferred environment manager (like Conda).

4.  **Install dependencies:**

    - Using `uv` (recommended):
      ```bash
      uv pip install -r requirements.txt
      ```
    - Or using `pip`:
      ```bash
      pip install -r requirements.txt
      ```
    - _Note:_ If you encounter issues with `JSONLoader`, ensure `jq` is installed: `uv pip install jq` or `pip install jq`.

5.  **Configure Environment Variables:**

    - Copy the example environment file:
      ```bash
      cp .env.example .env
      ```
    - Edit the `.env` file:
      - Uncomment and set `EMBEDDING_PROVIDER` and `LLM_PROVIDER` (e.g., `"ollama"`).
      - Fill in the necessary credentials/settings for your chosen provider(s) (e.g., `OLLAMA_EMBEDDING_MODEL`, `OLLAMA_LLM_MODEL`, `OPENAI_API_KEY`, etc.).

6.  **Ensure Data:** Place your source documents (`.txt`, `.json`, `.csv`) inside the `data/` directory.

7.  **(Optional) Pre-build Database:** You can manually build the database if needed:
    ```bash
    python load_db.py
    ```
    (The backend will also build it automatically on first run if it's missing).

## Running the Backend

1.  **Make sure your chosen LLM provider is running** (e.g., start the Ollama server if using Ollama).

2.  **Start the FastAPI server from the `backend` directory:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```
    The server will be accessible at `http://localhost:8000`. The API documentation (Swagger UI) is available at `http://localhost:8000/docs`.

## Vector Store Management

The application now supports multiple vector stores, with each embedding configuration getting its own isolated database. This eliminates compatibility issues when switching between different embedding models.

### Using the CLI Management Tool

The `manage_stores.py` script provides command-line tools for managing vector stores:

**List all vector stores:**

```bash
cd backend
python manage_stores.py list
```

**Show current embedding configuration:**

```bash
python manage_stores.py current
```

**Delete a specific vector store:**

```bash
python manage_stores.py delete <store-identifier>
```

**Delete all vector stores:**

```bash
python manage_stores.py delete-all
```

**Rebuild the current vector store:**

```bash
python manage_stores.py rebuild
```

### Using the API Endpoints

**List all vector stores:**

```bash
curl -X GET "http://localhost:8000/vector-stores"
```

**Delete a specific vector store:**

```bash
curl -X DELETE "http://localhost:8000/vector-stores/<store-identifier>"
```

**Delete all vector stores:**

```bash
curl -X DELETE "http://localhost:8000/vector-stores"
```

**Rebuild current vector store:**

```bash
curl -X POST "http://localhost:8000/vector-stores/rebuild" \\
  -H "Content-Type: application/json" \\
  -d '{"force": true}'
```

### Switching Embedding Models

To switch to a different embedding model:

1. Update your `.env` file with the new embedding provider and model
2. Restart the server
3. The system will automatically create a new vector store for the new configuration
4. Your old vector stores remain available and can be used by switching back to the previous configuration

**Example:** Switching from OpenAI to Ollama embeddings:

```bash
# In .env file
EMBEDDING_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434
```

The old OpenAI vector store will be preserved, and a new Ollama vector store will be created automatically.

## API Usage

### Main Query Endpoint

Send a POST request to the `/ask` endpoint with your question and a unique `session_id` in the JSON body. The `session_id` is used to maintain conversational context.

**Example using `curl`:**

```bash
curl -X 'POST' \\
  'http://localhost:8000/ask' \\
  -H 'accept: application/json' \\
  -H 'Content-Type: application/json' \\
  -d '{
  "question": "What were the main points discussed about Zilker Park?",
  "session_id": "user123_conversation456"
}'
```

**Example Response:**

```json
{
  "answer": "The Austin City Council discussed proposed funding for improvements to Zilker Park, including new trail maintenance and facility upgrades. Public comment period is now open.",
  "sources": [
    {
      "page_content": "[{\\"source\\": \\"Local News Outlet A\\", ...}]",
      "metadata": {
        "source": "/path/to/ai_agent_app/data/local_news.json",
        "seq_num": 1
      }
    }
    // ... other relevant sources might appear here
  ],
  "session_id": "user123_conversation456"
}
```

### Vector Store Management Endpoints

**GET /vector-stores** - List all available vector stores

```bash
curl -X GET "http://localhost:8000/vector-stores"
```

**DELETE /vector-stores/{identifier}** - Delete a specific vector store

```bash
curl -X DELETE "http://localhost:8000/vector-stores/openai_text-embedding-ada-002"
```

**DELETE /vector-stores** - Delete all vector stores

```bash
curl -X DELETE "http://localhost:8000/vector-stores"
```

**POST /vector-stores/rebuild** - Rebuild current vector store

```bash
curl -X POST "http://localhost:8000/vector-stores/rebuild" \\
  -H "Content-Type: application/json" \\
  -d '{"force": true}'
```

**GET /health** - Health check endpoint

```bash
curl -X GET "http://localhost:8000/health"
```

## Acknowledgments

- [Langchain](https://www.langchain.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
- [ChromaDB](https://www.trychroma.com/)
- [Ollama](https://ollama.com/)
- [LM Studio](https://lmstudio.ai/)
- [OpenAI](https://openai.com/)
- [Azure AI Services](https://azure.microsoft.com/en-us/products/ai-services/)
- [AWS Bedrock](https://aws.amazon.com/bedrock/)
- [HuggingFace Transformers & Embeddings](https://huggingface.co/)
- [Pydantic](https://docs.pydantic.dev/)
- [python-dotenv](https://github.com/theskumar/python-dotenv)
- Standard Python Libraries (`logging`, `os`, `pathlib`, `shutil`, `argparse`, `traceback`, `contextlib`)

## Troubleshooting

### Common Issues

**Vector Database Compatibility:**

- If you encounter vector database errors after switching embedding models, the system now automatically creates separate stores for each configuration
- Use `python manage_stores.py list` to see all available stores
- Use `python manage_stores.py current` to verify your current configuration

**Missing Vector Store:**

- If no vector store exists for your current configuration, the system will automatically create one on startup
- Ensure your `data/` directory contains documents to process
- Check logs for any document loading errors

**Memory Issues:**

- Large document collections may require more memory
- Consider chunking large documents or processing them in batches
- Monitor memory usage during vector store creation

**Provider Connection Issues:**

- Ensure your LLM provider service is running (e.g., Ollama, LM Studio)
- Verify API keys and endpoints in your `.env` file
- Check network connectivity to external providers (OpenAI, Azure, Bedrock)

### Log Analysis

The application provides detailed logging. Check the console output for:

- Vector store creation and loading messages
- Document processing statistics
- Provider connection status
- Error details with stack traces

### Vector Store Recovery

If you need to rebuild a corrupted vector store:

1. **Using CLI:**

   ```bash
   python manage_stores.py rebuild
   ```

2. **Using API:**

   ```bash
   curl -X POST "http://localhost:8000/vector-stores/rebuild" -H "Content-Type: application/json" -d '{"force": true}'
   ```

3. **Manual cleanup:**
   ```bash
   python manage_stores.py delete <store-identifier>
   # Restart server to auto-create new store
   ```

- If you encounter issues with the database or LLM provider, check the logs for error messages.
- Ensure that your environment variables are correctly set and that the required services are running.
- If you have questions or need help, feel free to open an issue in the repository.
- For any bugs or feature requests, please create an issue on the GitHub repository.
- For any questions or discussions, feel free to reach out on the project's GitHub Discussions page.

## Future Improvements

- **Enhanced Vector Store Management:** Add vector store versioning, automatic cleanup of old stores, and store compression.
- **Advanced Multi-Model Support:** Support for hybrid retrieval using multiple embedding models simultaneously.
- **Performance Optimization:** Implement caching for frequently asked questions and optimize vector search performance.
- **Production Features:** Add user authentication, rate limiting, and persistent chat history storage.
- **Monitoring & Analytics:** Add metrics collection, query performance tracking, and usage analytics.
- **Document Processing:** Support for more document formats, automatic document updates, and incremental indexing.
- **Deployment Options:** Docker containers, cloud deployment guides, and auto-scaling configurations.
- **Testing & Quality:** Comprehensive unit tests, integration tests, and automated testing pipelines.
- **Frontend Integration:** Web-based management interface for vector stores and chat interactions.
- **Enterprise Features:** Multi-tenancy support, advanced security, and audit logging.

### Recently Added Features ✨

- **Multi-Vector-Store Architecture:** Each embedding model now gets its own isolated vector database
- **CLI Management Tools:** Command-line utilities for listing, deleting, and managing vector stores
- **API Management Endpoints:** RESTful endpoints for programmatic vector store management
- **Automatic Store Creation:** Seamless switching between embedding models with automatic store creation
- **Metadata Tracking:** Detailed metadata for each vector store including creation time, document counts, and usage statistics
