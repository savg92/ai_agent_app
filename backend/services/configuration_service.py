"""Configuration helper for embedding configurations."""

import json
import logging
import os
from typing import Dict, Any, Optional
from config import get_settings, get_paths
from core.types import EmbeddingConfig


class ConfigurationService:
    """Service for managing embedding configurations."""
    
    def __init__(self):
        self.settings = get_settings()
        self.paths = get_paths()
    
    def get_current_embedding_config(self) -> EmbeddingConfig:
        """Get current embedding configuration for comparison."""
        provider = self.settings.embedding_provider
        config: EmbeddingConfig = {"provider": provider}
        
        if provider == "openai":
            config["model"] = "text-embedding-ada-002"  # Default OpenAI model
        elif provider == "ollama":
            config["model"] = self.settings.ollama_embedding_model or ""
            config["base_url"] = self.settings.ollama_base_url
        elif provider == "azure":
            config["deployment"] = self.settings.azure_embedding_deployment or ""
            config["endpoint"] = self.settings.azure_endpoint or ""
        elif provider == "bedrock":
            config["model_id"] = self.settings.bedrock_embedding_model_id or ""
            config["region"] = self.settings.aws_region or ""
        elif provider == "lmstudio":
            config["model"] = self.settings.lm_studio_embedding_model or ""
            config["base_url"] = self.settings.lm_studio_base_url
        else:
            config["model"] = "all-MiniLM-L6-v2"  # Default HuggingFace model
        
        return config
    
    def save_embedding_config(self, config: EmbeddingConfig) -> None:
        """Save current embedding configuration to file."""
        try:
            with open(self.paths.embedding_config_file, 'w') as f:
                json.dump(dict(config), f, indent=2)
            logging.debug(f"Saved embedding config to {self.paths.embedding_config_file}")
        except Exception as e:
            logging.warning(f"Failed to save embedding config: {e}")
    
    def load_saved_embedding_config(self) -> Optional[EmbeddingConfig]:
        """Load previously saved embedding configuration."""
        try:
            if os.path.exists(self.paths.embedding_config_file):
                with open(self.paths.embedding_config_file, 'r') as f:
                    config_data = json.load(f)
                logging.debug(f"Loaded embedding config from {self.paths.embedding_config_file}")
                return config_data
        except Exception as e:
            logging.warning(f"Failed to load embedding config: {e}")
        return None
    
    def has_embedding_config_changed(self) -> bool:
        """Check if embedding configuration has changed since last run."""
        current_config = self.get_current_embedding_config()
        saved_config = self.load_saved_embedding_config()
        
        if saved_config is None:
            logging.info("No previous embedding config found")
            return True
        
        if current_config != saved_config:
            logging.info(f"Embedding config changed from {saved_config} to {current_config}")
            return True
        
        logging.debug("Embedding config unchanged")
        return False
