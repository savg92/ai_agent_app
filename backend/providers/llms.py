"""LLM provider factory and implementations."""

import logging
from typing import Dict, Any
from langchain_core.language_models.llms import BaseLanguageModel
from langchain_openai import OpenAI, ChatOpenAI, AzureChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_community.llms import Bedrock

from config import get_settings


class LLMProviderFactory:
    """Factory class for creating LLM providers."""
    
    @staticmethod
    def create_llm(provider: str = None) -> BaseLanguageModel:
        """Create an LLM based on the provider."""
        settings = get_settings()
        provider = provider or settings.llm_provider
        provider = provider.lower()
        
        logging.info(f"Using LLM provider: {provider}")
        temperature = settings.llm_temperature

        if provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables.")
            return OpenAI(api_key=settings.openai_api_key, temperature=temperature)
        
        elif provider == "ollama":
            if not settings.ollama_llm_model:
                raise ValueError("OLLAMA_LLM_MODEL not found in environment variables for Ollama LLM provider.")
            logging.info(f"Using Ollama LLM model: {settings.ollama_llm_model} at {settings.ollama_base_url}")
            return OllamaLLM(
                model=settings.ollama_llm_model, 
                base_url=settings.ollama_base_url, 
                temperature=temperature
            )
        
        elif provider == "azure":
            required_vars = [
                settings.azure_api_key,
                settings.azure_endpoint,
                settings.azure_api_version,
                settings.azure_llm_deployment
            ]
            if not all(required_vars):
                raise ValueError("Missing required Azure OpenAI configuration (Key, Endpoint, Version, LLM Deployment Name) in environment variables.")
            
            logging.info(f"Using Azure LLM deployment: {settings.azure_llm_deployment}")
            return AzureChatOpenAI(
                api_key=settings.azure_api_key,
                azure_endpoint=settings.azure_endpoint,
                api_version=settings.azure_api_version,
                azure_deployment=settings.azure_llm_deployment,
                temperature=temperature
            )
        
        elif provider == "bedrock":
            if not settings.bedrock_model_id:
                raise ValueError("BEDROCK_MODEL_ID not found in environment variables for Bedrock provider.")
            
            if not settings.aws_region:
                logging.warning("AWS_REGION not set for Bedrock LLM, it might default or fail.")

            logging.info(f"Using Bedrock model: {settings.bedrock_model_id} in region {settings.aws_region or 'default'}")
            
            bedrock_params: Dict[str, Any] = {"model_id": settings.bedrock_model_id}
            
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

            bedrock_params["model_kwargs"] = {"temperature": temperature}
            return Bedrock(**bedrock_params)
        
        elif provider == "lmstudio":
            if not settings.lm_studio_model:
                raise ValueError("LM_STUDIO_MODEL not found in environment variables for LM Studio provider.")
            
            logging.info(f"Using LM Studio model: {settings.lm_studio_model} at {settings.lm_studio_base_url}")
            return ChatOpenAI(
                api_key=settings.lm_studio_api_key,
                base_url=f"{settings.lm_studio_base_url}/v1",
                model=settings.lm_studio_model,
                temperature=temperature
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
