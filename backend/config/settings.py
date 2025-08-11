"""Application settings and configuration management."""

import os
from typing import Optional
from dotenv import load_dotenv


class Settings:
    """Application settings loaded from environment variables."""
    
    def __init__(self):
        load_dotenv()
        
        # Core settings
        self.app_name = "AI Agent Developer Test API"
        self.app_version = "1.0.0"
        self.app_description = "API for querying documents using RAG."
        
        # Server settings
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))
        
        # CORS settings
        self.cors_origins = [
            "http://localhost:3000",
            "http://localhost:5173",
        ]
        
        # Provider settings
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        
        # LLM settings
        self.llm_temperature = self._get_float_env("LLM_TEMPERATURE", 0.7, 0.0, 2.0)
        self.retriever_k = self._get_int_env("RETRIEVER_K", 3, min_val=1)
        self.retriever_search_type = os.getenv("RETRIEVER_SEARCH_TYPE", "similarity").lower()
        self.retriever_score_threshold = self._get_float_env("RETRIEVER_SCORE_THRESHOLD", 0.2, 0.0, 1.0)
        self.chat_history_max_turns = self._get_int_env("CHAT_HISTORY_MAX_TURNS", 8, min_val=0)
        self.embedding_batch_size = self._get_int_env("EMBEDDING_BATCH_SIZE", 64, min_val=1)
        
        # OpenAI settings
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Ollama settings
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL")
        self.ollama_llm_model = os.getenv("OLLAMA_LLM_MODEL")
        
        # Azure OpenAI settings
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        self.azure_llm_deployment = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT_NAME")
        
        # AWS Bedrock settings
        self.bedrock_embedding_model_id = os.getenv("BEDROCK_EMBEDDING_MODEL_ID")
        self.bedrock_model_id = os.getenv("BEDROCK_MODEL_ID")
        self.aws_region = os.getenv("AWS_REGION")
        self.aws_profile = os.getenv("BEDROCK_PROFILE_NAME")
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        # LM Studio settings
        self.lm_studio_base_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")
        self.lm_studio_api_key = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
        self.lm_studio_embedding_model = os.getenv("LM_STUDIO_EMBEDDING_MODEL")
        self.lm_studio_model = os.getenv("LM_STUDIO_MODEL")
        
        # Telemetry settings
        os.environ['ANONYMIZED_TELEMETRY'] = 'False'

    # ---- Runtime update helpers (non-persistent) ----
    def update_llm_settings(self, provider: str, **kwargs) -> None:
        """Update LLM provider and related settings at runtime without process restart.
        Values are held in-memory; env vars are not modified."""
        if not provider:
            return
        self.llm_provider = provider.lower()

        # Common knobs
        self.llm_temperature = kwargs.get('llm_temperature', self.llm_temperature)

        # Provider-specific fields (optional; only update if provided)
        # OpenAI
        if 'openai_api_key' in kwargs:
            self.openai_api_key = kwargs['openai_api_key']

        # Ollama
        if 'ollama_llm_model' in kwargs:
            self.ollama_llm_model = kwargs['ollama_llm_model']
        if 'ollama_base_url' in kwargs:
            self.ollama_base_url = kwargs['ollama_base_url']

        # Azure OpenAI
        if 'azure_api_key' in kwargs:
            self.azure_api_key = kwargs['azure_api_key']
        if 'azure_endpoint' in kwargs:
            self.azure_endpoint = kwargs['azure_endpoint']
        if 'azure_api_version' in kwargs:
            self.azure_api_version = kwargs['azure_api_version']
        if 'azure_llm_deployment' in kwargs:
            self.azure_llm_deployment = kwargs['azure_llm_deployment']

        # Bedrock
        if 'bedrock_model_id' in kwargs:
            self.bedrock_model_id = kwargs['bedrock_model_id']
        if 'aws_region' in kwargs:
            self.aws_region = kwargs['aws_region']
        if 'aws_profile' in kwargs:
            self.aws_profile = kwargs['aws_profile']
        if 'aws_access_key_id' in kwargs:
            self.aws_access_key_id = kwargs['aws_access_key_id']
        if 'aws_secret_access_key' in kwargs:
            self.aws_secret_access_key = kwargs['aws_secret_access_key']

        # LM Studio
        if 'lm_studio_model' in kwargs:
            self.lm_studio_model = kwargs['lm_studio_model']
        if 'lm_studio_base_url' in kwargs:
            self.lm_studio_base_url = kwargs['lm_studio_base_url']
        if 'lm_studio_api_key' in kwargs:
            self.lm_studio_api_key = kwargs['lm_studio_api_key']

    def current_llm_config(self) -> dict:
        """Return a sanitized snapshot of current LLM config (no secrets leaked)."""
        data = {
            'provider': self.llm_provider,
        }
        if self.llm_provider == 'openai':
            data['model'] = 'gpt-*'
        elif self.llm_provider == 'ollama':
            data['ollama_llm_model'] = self.ollama_llm_model
            data['ollama_base_url'] = self.ollama_base_url
        elif self.llm_provider == 'azure':
            data['azure_endpoint'] = self.azure_endpoint
            data['azure_api_version'] = self.azure_api_version
            data['azure_llm_deployment'] = self.azure_llm_deployment
        elif self.llm_provider == 'bedrock':
            data['bedrock_model_id'] = self.bedrock_model_id
            data['aws_region'] = self.aws_region
        elif self.llm_provider == 'lmstudio':
            data['lm_studio_model'] = self.lm_studio_model
            data['lm_studio_base_url'] = self.lm_studio_base_url
        return data

    # Embeddings runtime controls
    def update_embedding_settings(self, provider: str, **kwargs) -> None:
        """Update embedding provider and related settings at runtime (in-memory)."""
        if not provider:
            return
        self.embedding_provider = provider.lower()

        # OpenAI
        if 'openai_api_key' in kwargs:
            self.openai_api_key = kwargs['openai_api_key']

        # Ollama
        if 'ollama_embedding_model' in kwargs:
            self.ollama_embedding_model = kwargs['ollama_embedding_model']
        if 'ollama_base_url' in kwargs:
            self.ollama_base_url = kwargs['ollama_base_url']

        # Azure
        if 'azure_api_key' in kwargs:
            self.azure_api_key = kwargs['azure_api_key']
        if 'azure_endpoint' in kwargs:
            self.azure_endpoint = kwargs['azure_endpoint']
        if 'azure_api_version' in kwargs:
            self.azure_api_version = kwargs['azure_api_version']
        if 'azure_embedding_deployment' in kwargs:
            self.azure_embedding_deployment = kwargs['azure_embedding_deployment']

        # Bedrock
        if 'bedrock_embedding_model_id' in kwargs:
            self.bedrock_embedding_model_id = kwargs['bedrock_embedding_model_id']
        if 'aws_region' in kwargs:
            self.aws_region = kwargs['aws_region']
        if 'aws_profile' in kwargs:
            self.aws_profile = kwargs['aws_profile']
        if 'aws_access_key_id' in kwargs:
            self.aws_access_key_id = kwargs['aws_access_key_id']
        if 'aws_secret_access_key' in kwargs:
            self.aws_secret_access_key = kwargs['aws_secret_access_key']

        # LM Studio
        if 'lm_studio_embedding_model' in kwargs:
            self.lm_studio_embedding_model = kwargs['lm_studio_embedding_model']
        if 'lm_studio_base_url' in kwargs:
            self.lm_studio_base_url = kwargs['lm_studio_base_url']
        if 'lm_studio_api_key' in kwargs:
            self.lm_studio_api_key = kwargs['lm_studio_api_key']

    def current_embedding_config(self) -> dict:
        """Return the current embedding configuration for vector store identification."""
        data = {
            'provider': self.embedding_provider,
        }
        if self.embedding_provider == 'openai':
            data['model'] = 'text-embedding-ada-002'
        elif self.embedding_provider == 'ollama':
            data['model'] = self.ollama_embedding_model
            data['base_url'] = self.ollama_base_url
        elif self.embedding_provider == 'azure':
            data['deployment'] = self.azure_embedding_deployment
            data['endpoint'] = self.azure_endpoint
        elif self.embedding_provider == 'bedrock':
            data['model_id'] = self.bedrock_embedding_model_id
            data['region'] = self.aws_region
        elif self.embedding_provider == 'lmstudio':
            data['model'] = self.lm_studio_embedding_model
            data['base_url'] = self.lm_studio_base_url
        else:
            data['model'] = 'all-MiniLM-L6-v2'
        return data
    
    def _get_float_env(self, key: str, default: float, min_val: float = None, max_val: float = None) -> float:
        """Get float environment variable with validation."""
        try:
            value = float(os.getenv(key, str(default)))
            if min_val is not None and value < min_val:
                return default
            if max_val is not None and value > max_val:
                return default
            return value
        except (ValueError, TypeError):
            return default
    
    def _get_int_env(self, key: str, default: int, min_val: int = None, max_val: int = None) -> int:
        """Get integer environment variable with validation."""
        try:
            value = int(os.getenv(key, str(default)))
            if min_val is not None and value < min_val:
                return default
            if max_val is not None and value > max_val:
                return default
            return value
        except (ValueError, TypeError):
            return default


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
