# AI Agent RAG Application

This project implements a Retrieval-Augmented Generation (RAG) system using a Python backend built with FastAPI and Langchain. It allows users to ask questions about documents stored locally, leveraging various LLM and embedding providers like Ollama, OpenAI, Azure, Bedrock, and LM Studio.

## Features

*   **RAG Pipeline:** Retrieves relevant document chunks from a vector database (ChromaDB) and uses a Language Model (LLM) to generate answers based on the retrieved context.
*   **Conversational Memory:** Maintains chat history per session, allowing for follow-up questions.
*   **FastAPI Backend:** Provides an API endpoint (`/ask`) to interact with the RAG system.
*   **Multiple Provider Support:** Configurable to use different providers for embeddings (OpenAI, Ollama, Azure, Bedrock, HuggingFace, LM Studio) and LLMs (OpenAI, Ollama, Azure, Bedrock, LM Studio) via environment variables.
*   **Automatic Database Creation:** The backend automatically creates the vector database on first startup if it doesn't exist.
*   **CORS Enabled:** Allows requests from specified frontend origins (e.g., `http://localhost:3000`, `http://localhost:5173`).

## Project Structure

```
.
├── README.md
├── backend/
│   ├── main.py         # FastAPI application, API endpoints, startup logic
│   ├── utils.py        # Core logic for loading data, embeddings, LLMs, QA chain
│   ├── load_db.py      # (Optional) Script to manually load/reload the vector DB
│   ├── requirements.txt # Python dependencies
│   ├── .env.example    # Example environment variable configuration
│   ├── .env            # Local environment variables (ignored by git)
│   └── chroma_db_store/ # Generated vector database (ignored by git)
├── data/
│   ├── local_news.json # Example data file
│   └── tax_policies.txt # Example data file
└── frontend/           # (Placeholder for frontend application)
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
    *   Using `venv`:
        ```bash
        python -m venv .venv
        source .venv/bin/activate
        ```
    *   Using `uv` (recommended):
        ```bash
        uv venv .venv
        source .venv/bin/activate
        ```
    *   On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```
    *   On Windows:
        ```bash
        .venv\Scripts\activate
        ```
    *   *Note:* If you encounter issues with `uv`, you can use `pip` to create a virtual environment:
        ```bash
        python -m venv .venv
        source .venv/bin/activate
        ```
    *   *Note:* If you are using a different environment manager (like Conda), create and activate your environment accordingly:
        ```bash
        conda create -n myenv python=3.9
        conda activate myenv
        ```

    *   Or using your preferred environment manager (like Conda).

4.  **Install dependencies:**
    *   Using `uv` (recommended):
        ```bash
        uv pip install -r requirements.txt
        ```
    *   Or using `pip`:
        ```bash
        pip install -r requirements.txt
        ```
    *   *Note:* If you encounter issues with `JSONLoader`, ensure `jq` is installed: `uv pip install jq` or `pip install jq`.

5.  **Configure Environment Variables:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file:
        *   Uncomment and set `EMBEDDING_PROVIDER` and `LLM_PROVIDER` (e.g., `"ollama"`).
        *   Fill in the necessary credentials/settings for your chosen provider(s) (e.g., `OLLAMA_EMBEDDING_MODEL`, `OLLAMA_LLM_MODEL`, `OPENAI_API_KEY`, etc.).

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

## API Usage

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


## Acknowledgments
*   [Langchain](https://www.langchain.com/)
*   [FastAPI](https://fastapi.tiangolo.com/)
*   [Uvicorn](https://www.uvicorn.org/)
*   [ChromaDB](https://www.trychroma.com/)
*   [Ollama](https://ollama.com/)
*   [LM Studio](https://lmstudio.ai/)
*   [OpenAI](https://openai.com/)
*   [Azure AI Services](https://azure.microsoft.com/en-us/products/ai-services/)
*   [AWS Bedrock](https://aws.amazon.com/bedrock/)
*   [HuggingFace Transformers & Embeddings](https://huggingface.co/)
*   [Pydantic](https://docs.pydantic.dev/)
*   [python-dotenv](https://github.com/theskumar/python-dotenv)
*   Standard Python Libraries (`logging`, `os`, `pathlib`, `shutil`, `argparse`, `traceback`, `contextlib`)

## Troubleshooting
*   If you encounter issues with the database or LLM provider, check the logs for error messages.
*   Ensure that your environment variables are correctly set and that the required services are running.
*   If you have questions or need help, feel free to open an issue in the repository.
*   For any bugs or feature requests, please create an issue on the GitHub repository.
*   For any questions or discussions, feel free to reach out on the project's GitHub Discussions page.

## Future Improvements
*   Add more detailed error handling and logging.
*   Support for more document formats and data sources.
*   Add user authentication and authorization for the API.
*   Implement caching for frequently asked questions to improve performance.
*   Explore additional LLM and embedding providers for more flexibility.
*   Add unit tests and integration tests for the backend.
*   Improve documentation and examples for easier onboarding.
*   Explore deployment options (e.g., Docker, cloud services) for easier distribution and scaling.
*   Consider replacing the in-memory chat history with a persistent solution (e.g., Redis, database) for production use.
*   Explore integration with other AI tools and platforms for enhanced capabilities.
*   Consider adding a frontend application for a better user experience.