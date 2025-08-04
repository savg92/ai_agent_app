"""Embedding provider factory and implementations."""

import logging
import requests
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import BedrockEmbeddings, HuggingFaceEmbeddings

from config import get_settings


class LMStudioEmbeddings(Embeddings):
    """Custom embedding class for LM Studio that handles the API format correctly."""
    
    def __init__(self, base_url: str, model: str, api_key: str = "lm-studio"):
        self.base_url = base_url.rstrip('/') + '/v1'
        self.model = model
        self.api_key = api_key
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        return self._get_embedding(text)
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text using LM Studio API."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "input": text,  # LM Studio expects 'input' as a string
            "model": self.model
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            if 'data' in result and len(result['data']) > 0:
                return result['data'][0]['embedding']
            else:
                raise ValueError(f"Unexpected response format from LM Studio: {result}")
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Error calling LM Studio embedding API: {e}")
            raise ValueError(f"Failed to get embedding from LM Studio: {e}")
        except (KeyError, IndexError) as e:
            logging.error(f"Error parsing LM Studio response: {e}")
            raise ValueError(f"Invalid response format from LM Studio: {e}")


class EmbeddingProviderFactory:
    """Factory class for creating embedding providers."""
    
    @staticmethod
    def create_embedding_function(provider: str = None) -> Embeddings:
        """Create an embedding function based on the provider."""
        settings = get_settings()
        provider = provider or settings.embedding_provider
        provider = provider.lower()
        
        logging.info(f"Using embedding provider: {provider}")

        if provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables.")
            return OpenAIEmbeddings(api_key=settings.openai_api_key)

        elif provider == "ollama":
            if not settings.ollama_embedding_model:
                raise ValueError("OLLAMA_EMBEDDING_MODEL not found in environment variables for Ollama embedding provider.")
            logging.info(f"Using Ollama embedding model: {settings.ollama_embedding_model} at {settings.ollama_base_url}")
            return OllamaEmbeddings(
                model=settings.ollama_embedding_model, 
                base_url=settings.ollama_base_url
            )

        elif provider == "azure":
            required_vars = [
                settings.azure_api_key,
                settings.azure_endpoint,
                settings.azure_api_version,
                settings.azure_embedding_deployment
            ]
            if not all(required_vars):
                raise ValueError("Missing required Azure OpenAI configuration (Key, Endpoint, Version, Embedding Deployment Name) in environment variables.")
            
            logging.info(f"Using Azure embedding deployment: {settings.azure_embedding_deployment}")
            return AzureOpenAIEmbeddings(
                api_key=settings.azure_api_key,
                azure_endpoint=settings.azure_endpoint,
                api_version=settings.azure_api_version,
                azure_deployment=settings.azure_embedding_deployment
            )

        elif provider == "bedrock":
            if not settings.bedrock_embedding_model_id:
                raise ValueError("BEDROCK_EMBEDDING_MODEL_ID not found in environment variables for Bedrock embedding provider.")
            
            if not settings.aws_region:
                logging.warning("AWS_REGION not set for Bedrock embeddings, it might default or fail.")

            logging.info(f"Using Bedrock embedding model: {settings.bedrock_embedding_model_id} in region {settings.aws_region or 'default'}")
            
            bedrock_params = {"model_id": settings.bedrock_embedding_model_id}
            
            if settings.aws_region:
                bedrock_params["region_name"] = settings.aws_region
            
            if settings.aws_profile:
                if settings.aws_access_key_id or settings.aws_secret_access_key:
                    logging.warning("Both BEDROCK_PROFILE_NAME and AWS access keys found in environment. Using profile name.")
                bedrock_params["credentials_profile_name"] = settings.aws_profile
            elif settings.aws_access_key_id and settings.aws_secret_access_key:
                logging.info("Using AWS Access Key ID and Secret Access Key for Bedrock authentication.")
                bedrock_params["aws_access_key_id"] = settings.aws_access_key_id
                bedrock_params["aws_secret_access_key"] = settings.aws_secret_access_key

            return BedrockEmbeddings(**bedrock_params)

        elif provider == "lmstudio":
            if not settings.lm_studio_embedding_model:
                raise ValueError("LM_STUDIO_EMBEDDING_MODEL not found in environment variables for LM Studio embedding provider.")
            
            logging.info(f"Using LM Studio embedding model: {settings.lm_studio_embedding_model} at {settings.lm_studio_base_url}")
            return LMStudioEmbeddings(
                base_url=settings.lm_studio_base_url,
                model=settings.lm_studio_embedding_model,
                api_key=settings.lm_studio_api_key
            )

        else:
            logging.info("Provider not explicitly supported or specified, defaulting to Sentence Transformers (all-MiniLM-L6-v2).")
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
