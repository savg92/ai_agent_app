# AI Agent RAG Application - Modular Architecture

This project implements a **Retrieval-Augmented Generation (RAG)** system using a modern, modular Python backend built with FastAPI and Langchain. It allows users to ask questions about documents stored locally, leveraging various LLM and embedding providers like Ollama, OpenAI, Azure, Bedrock, and LM Studio.

## 🏗️ **Modular Architecture**

The application has been completely refactored from a monolithic structure to a clean, modular architecture with clear separation of concerns:

```
backend/
├── config/                     # 📂 Configuration Management
│   ├── settings.py            # Environment variables & validation
│   └── paths.py              # Path management & directory structure
├── core/                      # 📂 Core Models & Types
│   ├── models.py             # Pydantic API request/response models
│   └── types.py              # Type definitions & TypedDicts
├── providers/                 # 📂 Provider Factories
│   ├── embeddings.py         # Embedding providers (OpenAI, Ollama, Azure, etc.)
│   └── llms.py              # LLM providers with factory pattern
├── services/                  # 📂 Business Logic Services
│   ├── document_service.py          # Document loading & text processing
│   ├── vector_store_service.py      # Vector store creation & management
│   ├── qa_service.py               # Question-answering chain logic
│   ├── vector_store_manager.py     # Multi-store management system
│   └── configuration_service.py    # Configuration persistence & comparison
├── api/                       # 📂 API Layer
│   └── routes.py             # FastAPI endpoints & HTTP handling
├── main.py                    # 🚀 Application entry point
├── vector_stores/             # 📂 Multi-Store Vector Databases
│   ├── openai_text-embedding-ada-002/
│   ├── ollama_granite-embedding_278m/
│   └── lmstudio_text-embedding-model/
└── data/                      # 📂 Source Documents
    ├── local_news.json
    └── tax_policies.txt
```

## ✨ **Key Features**

### 🎯 **Modern Architecture Benefits**

- **Separation of Concerns**: Each module has a single, clear responsibility
- **Testability**: Individual services can be tested in isolation
- **Maintainability**: Changes to one module don't affect others
- **Extensibility**: Easy to add new providers or services
- **Reusability**: Services can be reused across different contexts

### 🔥 **Core Functionality**

- **RAG Pipeline**: Retrieves relevant document chunks and generates contextual answers
- **Multi-Vector-Store Architecture**: Isolated databases for each embedding configuration
- **Conversational Memory**: Session-based chat history with follow-up support
- **FastAPI Backend**: Modern async API with automatic OpenAPI documentation
- **Multiple Provider Support**: Configurable LLM and embedding providers
- **Automatic Database Creation**: Smart initialization and model switching
- **Vector Store Management**: Complete API and CLI management tools
- **CORS Enabled**: Frontend-ready with configurable origins

## 🚀 **Quick Start**

### 1. **Setup Environment**

```bash
# Clone and navigate
git clone https://github.com/savg92/ai_agent_app
cd ai_agent_app/backend

# Create environment (choose one)
uv venv .venv              # Using uv (recommended)
python -m venv .venv       # Using standard Python

# Activate environment
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate     # On Windows
```

### 2. **Install Dependencies**

```bash
# Modern approach (recommended)
uv pip install -r requirements.txt

# Or traditional approach
pip install -r requirements.txt
```

### 3. **Configure Environment**

```bash
# Copy and edit configuration
cp .env.example .env
# Edit .env with your preferred providers and API keys
```

### 4. **Start the Server**

```bash
# From the backend directory
uv run python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Or using Python directly
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server will be accessible at `http://localhost:8000` with API docs at `http://localhost:8000/docs`.

## ⚙️ **Configuration**

The modular architecture centralizes all configuration in the `config/` module. All settings are loaded from environment variables with validation and type conversion.

### **Provider Configuration**

**OpenAI:**

```bash
EMBEDDING_PROVIDER=openai
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
```

**Ollama:**

```bash
EMBEDDING_PROVIDER=ollama
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=granite-embedding:278m
OLLAMA_LLM_MODEL=llama3.2:3b
```

**Azure OpenAI:**

```bash
EMBEDDING_PROVIDER=azure
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002
AZURE_OPENAI_LLM_DEPLOYMENT_NAME=gpt-35-turbo
```

**LM Studio:**

```bash
EMBEDDING_PROVIDER=lmstudio
LLM_PROVIDER=lmstudio
LM_STUDIO_BASE_URL=http://localhost:1234
LM_STUDIO_EMBEDDING_MODEL=text-embedding-qwen3-embedding-0.6b
LM_STUDIO_MODEL=qwen2.5-coder-7b-instruct
```

**AWS Bedrock:**

```bash
EMBEDDING_PROVIDER=bedrock
LLM_PROVIDER=bedrock
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v1
BEDROCK_MODEL_ID=anthropic.claude-v2
AWS_REGION=us-east-1
```

## 🔧 **API Usage**

### **Ask Questions**

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is this application about?",
    "session_id": "user-session-123"
  }'
```

### **Vector Store Management**

````bash
# List all vector stores
curl -X GET "http://localhost:8000/vector-stores"

# Delete a specific store
curl -X DELETE "http://localhost:8000/vector-stores/ollama_granite-embedding_278m"

# Rebuild current store
curl -X POST "http://localhost:8000/vector-stores/rebuild" \
  -H "Content-Type: application/json" \
  -d '{"force": true}'

# Health check
curl -X GET "http://localhost:8000/health"

### **Runtime LLM Switching**

You can change the LLM provider without restarting the server. The vector store and sessions remain intact.

```bash
# Inspect current LLM
curl -X GET "http://localhost:8000/llm"

# Switch to Ollama (example)
curl -X POST "http://localhost:8000/llm" \
    -H "Content-Type: application/json" \
    -d '{
        "provider": "ollama",
        "ollama_llm_model": "llama3.2:3b",
        "ollama_base_url": "http://localhost:11434",
        "llm_temperature": 0.7
    }'

# Switch to Azure OpenAI (example)
curl -X POST "http://localhost:8000/llm" \
    -H "Content-Type: application/json" \
    -d '{
        "provider": "azure",
        "azure_api_key": "<key>",
        "azure_endpoint": "https://<resource>.openai.azure.com/",
        "azure_api_version": "2024-02-01",
        "azure_llm_deployment": "gpt-4o-mini"
    }'
````

````

## 🧩 **Modular Development**

### **Adding a New Provider**

1. **Update Provider Factory** (`providers/embeddings.py` or `providers/llms.py`):

```python
elif provider == "new_provider":
    # Implementation here
    return NewProviderEmbeddings(...)
````

2. **Add Configuration** (`config/settings.py`):

```python
# Add new provider settings
self.new_provider_api_key = os.getenv("NEW_PROVIDER_API_KEY")
```

### **Creating a New Service**

1. **Create Service** (`services/new_service.py`):

```python
class NewService:
    def __init__(self):
        self.settings = get_settings()

    def do_something(self):
        # Implementation here
        pass
```

2. **Update Exports** (`services/__init__.py`):

```python
from .new_service import NewService
__all__.append("NewService")
```

### **Adding API Endpoints**

Add to `api/routes.py`:

```python
@router.get("/new-endpoint")
def new_endpoint():
    service = NewService()
    return service.do_something()
```

## 🧪 **Testing**

The modular structure makes testing straightforward:

```python
# Test individual services
def test_document_service():
    service = DocumentService()
    documents, failed = service.load_documents_from_directory("/path/to/docs")
    assert len(documents) > 0

# Test providers
def test_embedding_factory():
    embeddings = EmbeddingProviderFactory.create_embedding_function("openai")
    assert embeddings is not None
```

## 📊 **Multi-Vector Store System**

Each embedding configuration gets its own isolated vector database:

- **Automatic Creation**: New stores created when switching models
- **Metadata Tracking**: Creation time, document count, usage statistics
- **Easy Switching**: Change `.env` and restart - no data loss
- **CLI Management**: Full command-line management tools
- **API Management**: RESTful endpoints for programmatic control

## 🔍 **Migration from Legacy**

The application maintains **100% API compatibility** with the previous monolithic version. All existing endpoints work exactly the same way, but the internal implementation is now much cleaner and more maintainable.

**Migration artifacts:**

- `backup_original/` - Contains original monolithic files
- `MODULAR_ARCHITECTURE.md` - Detailed architecture documentation
- `MIGRATION_COMPLETE.md` - Migration summary and validation

## 🛠️ **Development Workflow**

1. **Make changes** to specific modules based on functionality
2. **Use dependency injection** to pass services between modules
3. **Keep API layer thin** - business logic belongs in services
4. **Add environment variables** in `config/settings.py`
5. **Update type definitions** in `core/types.py` as needed

## 📚 **Documentation**

- **`MODULAR_ARCHITECTURE.md`** - Complete architecture guide
- **`MIGRATION_COMPLETE.md`** - Migration summary and status
- **`CLEANUP_SUMMARY.md`** - File cleanup documentation
- **OpenAPI Docs** - Available at `http://localhost:8000/docs`

## 🚦 **Production Ready**

The modular architecture provides a solid foundation for production deployment:

- **Clean separation of concerns**
- **Easy testing and mocking**
- **Configurable via environment variables**
- **Comprehensive error handling**
- **Structured logging**
- **Type safety with Pydantic**
- **Async-ready with FastAPI**

## 🎯 **Future Development**

The modular structure makes it easy to:

- Add new LLM/embedding providers
- Implement caching layers
- Add authentication and authorization
- Scale individual components
- Add monitoring and metrics
- Implement advanced RAG techniques
- Create specialized services for different use cases

---

**The AI Agent RAG Application is now production-ready with a professional, maintainable modular architecture! 🎉**

## 🔗 **Acknowledgments**

Built with modern, professional tools and libraries:

- **Core Framework**: [FastAPI](https://fastapi.tiangolo.com/) - Modern, fast web framework for building APIs
- **AI/ML Stack**: [Langchain](https://www.langchain.com/) - Framework for developing LLM applications
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) - AI-native open-source embedding database
- **Server**: [Uvicorn](https://www.uvicorn.org/) - Lightning-fast ASGI server
- **Data Validation**: [Pydantic](https://docs.pydantic.dev/) - Data validation using Python type annotations

**Provider Integrations:**

- [OpenAI](https://openai.com/) - GPT models and embeddings
- [Ollama](https://ollama.com/) - Local LLM deployment platform
- [LM Studio](https://lmstudio.ai/) - Desktop application for running LLMs
- [Azure AI Services](https://azure.microsoft.com/en-us/products/ai-services/) - Microsoft's AI platform
- [AWS Bedrock](https://aws.amazon.com/bedrock/) - Amazon's managed AI service
- [HuggingFace](https://huggingface.co/) - Open-source ML models and embeddings

## 🐛 **Troubleshooting**

### **Common Issues**

**🔧 Server Won't Start:**

```bash
# Make sure you're in the backend directory
cd /Users/your-path/ai_agent_app/backend

# Use the correct python command
uv run python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**🔧 Provider Connection Issues:**

- Ensure your LLM service is running (Ollama, LM Studio, etc.)
- Verify API keys in your `.env` file
- Check network connectivity for external providers

**🔧 Vector Store Issues:**

- Use `curl -X GET "http://localhost:8000/vector-stores"` to list available stores
- Check that documents exist in the `data/` directory
- Verify embedding provider configuration

**🔧 Import/Module Errors:**

- Ensure all dependencies are installed: `uv pip install -r requirements.txt`
- Verify you're using the correct Python environment
- Check that all required packages are available

### **Getting Help**

- **API Documentation**: Visit `http://localhost:8000/docs` when server is running
- **Logs**: Check console output for detailed error messages and stack traces
- **Configuration**: Verify your `.env` file matches `.env.example` format
- **GitHub Issues**: Report bugs or request features in the repository

## 📈 **What's New in v2.0 (Modular Architecture)**

### ✨ **Major Improvements**

- **🏗️ Complete Modular Refactor**: Clean separation of concerns across 5 main modules
- **🧪 Enhanced Testability**: Each component can be tested independently
- **📦 Modern Architecture**: Industry-standard patterns with dependency injection
- **🔧 Better Maintainability**: Changes isolated to specific modules
- **🚀 Developer Experience**: Clear structure for adding new features

### 🔄 **Migration Benefits**

- **100% API Compatibility**: All existing endpoints work unchanged
- **Zero Downtime**: Seamless transition from monolithic structure
- **Preserved Data**: All vector stores and configurations maintained
- **Enhanced Performance**: More efficient service instantiation and management
- **Professional Codebase**: Production-ready architecture patterns

### 🛠️ **Development Enhancements**

- **Factory Patterns**: Clean provider instantiation and management
- **Service Layer**: Clear business logic separation
- **Configuration Management**: Centralized environment handling
- **Type Safety**: Comprehensive TypedDict and Pydantic integration
- **Error Handling**: Structured exception management across modules

---

**Ready to build the future of document intelligence with a rock-solid, modular foundation! 🚀✨**
