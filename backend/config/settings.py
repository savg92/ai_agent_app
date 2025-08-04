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
