import os
import logging
from typing import Optional, Dict, Any, List, Tuple
from dotenv import load_dotenv
import shutil
from langchain_community.document_loaders import UnstructuredFileLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document
from langchain.chains import ConversationalRetrievalChain
from langchain_core.language_models.llms import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.llms import Bedrock
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAI, ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_openai import AzureOpenAI, AzureChatOpenAI
import pathlib
import requests
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Disable Chroma telemetry
os.environ['ANONYMIZED_TELEMETRY'] = 'False'


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


BACKEND_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = BACKEND_DIR.parent.resolve()
CHROMA_DB_DIRECTORY = str(BACKEND_DIR / "chroma_db_store")
DATA_PATH = str(PROJECT_ROOT / "data")
EMBEDDING_CONFIG_FILE = str(BACKEND_DIR / "embedding_config.json")


def get_embedding_function(provider: str = os.getenv("EMBEDDING_PROVIDER", "openai")) -> Embeddings:
    provider = provider.lower()
    logging.info(f"Using embedding provider: {provider}")

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        return OpenAIEmbeddings(api_key=api_key)

    elif provider == "ollama":
        ollama_base_url: Optional[str] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_embedding_model: Optional[str] = os.getenv("OLLAMA_EMBEDDING_MODEL")
        if not ollama_embedding_model:
            logging.error("OLLAMA_EMBEDDING_MODEL not found in environment variables for Ollama embedding provider.")
            raise ValueError("OLLAMA_EMBEDDING_MODEL not found in environment variables for Ollama embedding provider.")
        logging.info(f"Using Ollama embedding model: {ollama_embedding_model} at {ollama_base_url}")
        return OllamaEmbeddings(model=ollama_embedding_model, base_url=ollama_base_url)

    elif provider == "azure":
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        if not all([azure_api_key, azure_endpoint, azure_api_version, azure_embedding_deployment]):
            logging.error("Missing required Azure OpenAI configuration (Key, Endpoint, Version, Embedding Deployment Name) in environment variables.")
            raise ValueError("Missing required Azure OpenAI configuration (Key, Endpoint, Version, Embedding Deployment Name) in environment variables.")
        logging.info(f"Using Azure embedding deployment: {azure_embedding_deployment}")
        return AzureOpenAIEmbeddings(
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            api_version=azure_api_version,
            azure_deployment=azure_embedding_deployment
        )

    elif provider == "bedrock":
        model_id: Optional[str] = os.getenv("BEDROCK_EMBEDDING_MODEL_ID")
        region_name: Optional[str] = os.getenv("AWS_REGION")
        credentials_profile_name: Optional[str] = os.getenv("BEDROCK_PROFILE_NAME")
        aws_access_key_id: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")

        if not model_id:
            logging.error("BEDROCK_EMBEDDING_MODEL_ID not found in environment variables for Bedrock embedding provider.")
            raise ValueError("BEDROCK_EMBEDDING_MODEL_ID not found in environment variables for Bedrock embedding provider.")
        if not region_name:
             logging.warning("AWS_REGION not set for Bedrock embeddings, it might default or fail.")

        logging.info(f"Using Bedrock embedding model: {model_id} in region {region_name or 'default'}")
        bedrock_params: Dict[str, Any] = {
            "model_id": model_id,
        }
        if region_name:
            bedrock_params["region_name"] = region_name
        if credentials_profile_name:
            if aws_access_key_id or aws_secret_access_key:
                logging.warning("Both BEDROCK_PROFILE_NAME and AWS access keys found in environment. Using profile name.")
            bedrock_params["credentials_profile_name"] = credentials_profile_name
        elif aws_access_key_id and aws_secret_access_key:
            logging.info("Using AWS Access Key ID and Secret Access Key for Bedrock authentication.")
            bedrock_params["aws_access_key_id"] = aws_access_key_id
            bedrock_params["aws_secret_access_key"] = aws_secret_access_key

        return BedrockEmbeddings(**bedrock_params)

    elif provider == "lmstudio":
        lm_studio_base_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")
        lm_studio_api_key = os.getenv("LM_STUDIO_API_KEY", "lm-studio")  # LM Studio uses a dummy key
        lm_studio_embedding_model = os.getenv("LM_STUDIO_EMBEDDING_MODEL")
        
        if not lm_studio_embedding_model:
            logging.error("LM_STUDIO_EMBEDDING_MODEL not found in environment variables for LM Studio embedding provider.")
            raise ValueError("LM_STUDIO_EMBEDDING_MODEL not found in environment variables for LM Studio embedding provider.")
        
        logging.info(f"Using LM Studio embedding model: {lm_studio_embedding_model} at {lm_studio_base_url}")
        return LMStudioEmbeddings(
            base_url=lm_studio_base_url,
            model=lm_studio_embedding_model,
            api_key=lm_studio_api_key
        )

    else:
        logging.info("Provider not explicitly supported or specified, defaulting to Sentence Transformers (all-MiniLM-L6-v2).")
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_current_embedding_config() -> Dict[str, Any]:
    """Get current embedding configuration for comparison."""
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    config = {"provider": provider}
    
    if provider == "openai":
        config["model"] = "text-embedding-ada-002"  # Default OpenAI model
    elif provider == "ollama":
        config["model"] = os.getenv("OLLAMA_EMBEDDING_MODEL", "")
        config["base_url"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    elif provider == "azure":
        config["deployment"] = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "")
        config["endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    elif provider == "bedrock":
        config["model_id"] = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "")
        config["region"] = os.getenv("AWS_REGION", "")
    elif provider == "lmstudio":
        config["model"] = os.getenv("LM_STUDIO_EMBEDDING_MODEL", "")
        config["base_url"] = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")
    else:
        config["model"] = "all-MiniLM-L6-v2"  # Default HuggingFace model
    
    return config


def save_embedding_config(config: Dict[str, Any]) -> None:
    """Save current embedding configuration to file."""
    try:
        with open(EMBEDDING_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        logging.debug(f"Saved embedding config to {EMBEDDING_CONFIG_FILE}")
    except Exception as e:
        logging.warning(f"Failed to save embedding config: {e}")


def load_saved_embedding_config() -> Optional[Dict[str, Any]]:
    """Load previously saved embedding configuration."""
    try:
        if os.path.exists(EMBEDDING_CONFIG_FILE):
            with open(EMBEDDING_CONFIG_FILE, 'r') as f:
                config = json.load(f)
            logging.debug(f"Loaded embedding config from {EMBEDDING_CONFIG_FILE}")
            return config
    except Exception as e:
        logging.warning(f"Failed to load embedding config: {e}")
    return None


def has_embedding_config_changed() -> bool:
    """Check if embedding configuration has changed since last run."""
    current_config = get_current_embedding_config()
    saved_config = load_saved_embedding_config()
    
    if saved_config is None:
        logging.info("No previous embedding config found")
        return True
    
    if current_config != saved_config:
        logging.info(f"Embedding config changed from {saved_config} to {current_config}")
        return True
    
    logging.debug("Embedding config unchanged")
    return False


def load_documents_from_directory(directory_path: str) -> Tuple[List[Document], List[str]]:
    documents: List[Document] = []
    failed_files: List[str] = []
    logging.info(f"Loading documents from directory: {directory_path}")
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if not os.path.isfile(file_path):
            continue

        try:
            if filename.endswith(".txt"):
                loader = UnstructuredFileLoader(file_path)
                documents.extend(loader.load())
            elif filename.endswith(".json"):
                loader = JSONLoader(file_path=file_path, jq_schema='.', text_content=False)
                loaded_json_docs = loader.load()
                logging.info(f"Loaded {len(loaded_json_docs)} document(s) from JSON file: {filename}")
                documents.extend(loaded_json_docs)
            elif filename.endswith(".csv"):
                loader = CSVLoader(file_path=file_path)
                documents.extend(loader.load())

            logging.debug(f"Loaded documents from: {filename}")
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}", exc_info=True)
            failed_files.append(file_path)

    logging.info(f"Finished loading documents. Total loaded: {len(documents)}. Failed files: {len(failed_files)}")
    if failed_files:
        logging.warning(f"Failed to load the following files: {failed_files}")
    return documents, failed_files

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\f", "\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    logging.info(f"Splitting {len(documents)} documents using separators: {text_splitter._separators}")
    split_docs: List[Document] = text_splitter.split_documents(documents)
    logging.info(f"Split into {len(split_docs)} chunks.")
    return split_docs


def create_or_load_vector_db(
    documents: Optional[List[Document]] = None,
    embedding_function: Optional[Embeddings] = None,
    force_reload: bool = False
) -> Optional[VectorStore]:
    if not embedding_function:
        embedding_function = get_embedding_function()

    persist_directory: str = CHROMA_DB_DIRECTORY
    vector_db: Optional[VectorStore] = None
    
    # Check if embedding configuration has changed
    config_changed = has_embedding_config_changed()
    if config_changed:
        logging.warning("Embedding configuration has changed. The existing vector database may be incompatible.")
        if os.path.exists(persist_directory):
            if not documents:
                logging.error("Embedding config changed but no documents provided to rebuild database.")
                raise ValueError("Embedding configuration changed. Please provide documents to rebuild the vector database or set force_reload=True.")
            logging.info("Automatically rebuilding vector database due to embedding config change...")
            force_reload = True

    if force_reload and documents:
        logging.info("Forcing reload of vector database...")
        if os.path.exists(persist_directory):
            logging.warning(f"*** WARNING: Removing existing database directory due to force_reload=True: {persist_directory}")
            shutil.rmtree(persist_directory)

        logging.info("Splitting documents...")
        texts: List[Document] = split_documents(documents)
        if not texts:
             logging.warning("No text chunks to add to the database after splitting.")
             return None
        logging.info(f"Creating new vector database with {len(texts)} chunks...")
        vector_db = Chroma.from_documents(
            documents=texts,
            embedding=embedding_function,
            persist_directory=persist_directory
        )
        vector_db.persist()
        logging.info(f"Vector database created and persisted at {persist_directory}")
        
        # Save the current embedding config after successful database creation
        current_config = get_current_embedding_config()
        save_embedding_config(current_config)
        
        return vector_db
    elif os.path.exists(persist_directory):
        logging.info(f"Loading existing vector database from {persist_directory}...")
        try:
            vector_db = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding_function
            )
            logging.info("Vector database loaded.")
            
            # Save current config if it hasn't been saved yet
            if config_changed:
                current_config = get_current_embedding_config()
                save_embedding_config(current_config)
                
            return vector_db
        except Exception as e:
            logging.error(f"Failed to load existing vector database: {e}")
            if documents:
                logging.info("Attempting to rebuild vector database...")
                shutil.rmtree(persist_directory)
                return create_or_load_vector_db(documents, embedding_function, force_reload=True)
            else:
                raise ValueError(f"Failed to load vector database and no documents provided for rebuild: {e}")
    elif documents:
         logging.info("No existing database found. Creating a new one...")
         logging.info("Splitting documents...")
         texts = split_documents(documents)
         if not texts:
             logging.warning("No text chunks to add to the database after splitting.")
             return None
         logging.info(f"Creating new vector database with {len(texts)} chunks...")
         vector_db = Chroma.from_documents(
            documents=texts,
            embedding=embedding_function,
            persist_directory=persist_directory
         )
         vector_db.persist()
         logging.info(f"Vector database created and persisted at {persist_directory}")
         
         # Save the current embedding config after successful database creation
         current_config = get_current_embedding_config()
         save_embedding_config(current_config)
         
         return vector_db
    else:
        logging.error("Cannot create or load vector DB: No documents provided and no existing database found.")
        return None


def get_llm(provider: str = os.getenv("LLM_PROVIDER", "openai")) -> BaseLanguageModel:
    provider = provider.lower()
    logging.info(f"Using LLM provider: {provider}")

    default_temp = 0.7
    try:
        temp_str = os.getenv("LLM_TEMPERATURE")
        if temp_str is None:
            temperature = default_temp
            logging.info(f"LLM_TEMPERATURE not set, using default: {temperature}")
        else:
            temperature = float(temp_str)
            if not (0.0 <= temperature <= 2.0):
                logging.warning(f"LLM_TEMPERATURE value ({temperature}) out of typical range [0.0, 2.0], using default: {default_temp}")
                temperature = default_temp
            else:
                logging.info(f"Using LLM_TEMPERATURE: {temperature}")
    except ValueError:
        logging.warning(f"LLM_TEMPERATURE value ('{temp_str}') is not a valid float, using default: {default_temp}")
        temperature = default_temp

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        return OpenAI(api_key=api_key, temperature=temperature)
    elif provider == "ollama":
        ollama_base_url: Optional[str] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_llm_model: Optional[str] = os.getenv("OLLAMA_LLM_MODEL")
        if not ollama_llm_model:
            logging.error("OLLAMA_LLM_MODEL not found in environment variables for Ollama LLM provider.")
            raise ValueError("OLLAMA_LLM_MODEL not found in environment variables for Ollama LLM provider.")
        logging.info(f"Using Ollama LLM model: {ollama_llm_model} at {ollama_base_url}")
        return OllamaLLM(model=ollama_llm_model, base_url=ollama_base_url, temperature=temperature)
    elif provider == "azure":
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        azure_llm_deployment = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT_NAME")
        if not all([azure_api_key, azure_endpoint, azure_api_version, azure_llm_deployment]):
            logging.error("Missing required Azure OpenAI configuration (Key, Endpoint, Version, LLM Deployment Name) in environment variables.")
            raise ValueError("Missing required Azure OpenAI configuration (Key, Endpoint, Version, LLM Deployment Name) in environment variables.")
        logging.info(f"Using Azure LLM deployment: {azure_llm_deployment}")
        return AzureChatOpenAI(
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            api_version=azure_api_version,
            azure_deployment=azure_llm_deployment,
            temperature=temperature
        )
    elif provider == "bedrock":
        model_id = os.getenv("BEDROCK_MODEL_ID")
        region_name = os.getenv("AWS_REGION")
        credentials_profile_name = os.getenv("BEDROCK_PROFILE_NAME")
        aws_access_key_id: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")

        if not model_id:
            logging.error("BEDROCK_MODEL_ID not found in environment variables for Bedrock provider.")
            raise ValueError("BEDROCK_MODEL_ID not found in environment variables for Bedrock provider.")
        if not region_name:
             logging.warning("AWS_REGION not set for Bedrock LLM, it might default or fail.")

        logging.info(f"Using Bedrock model: {model_id} in region {region_name or 'default'}")
        bedrock_params: Dict[str, Any] = {
            "model_id": model_id,
        }
        if region_name:
            bedrock_params["region_name"] = region_name
        if credentials_profile_name:
            if aws_access_key_id or aws_secret_access_key:
                logging.warning("Both BEDROCK_PROFILE_NAME and AWS access keys found in environment. Using profile name.")
            bedrock_params["credentials_profile_name"] = credentials_profile_name
        elif aws_access_key_id and aws_secret_access_key:
            logging.info("Using AWS Access Key ID and Secret Access Key for Bedrock authentication.")
            bedrock_params["aws_access_key_id"] = aws_access_key_id
            bedrock_params["aws_secret_access_key"] = aws_secret_access_key

        bedrock_params["model_kwargs"] = {"temperature": temperature}

        return Bedrock(**bedrock_params)
    elif provider == "lmstudio":
        lm_studio_base_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")
        lm_studio_api_key = os.getenv("LM_STUDIO_API_KEY", "lm-studio")  # LM Studio uses a dummy key
        lm_studio_model = os.getenv("LM_STUDIO_MODEL")
        
        if not lm_studio_model:
            logging.error("LM_STUDIO_MODEL not found in environment variables for LM Studio provider.")
            raise ValueError("LM_STUDIO_MODEL not found in environment variables for LM Studio provider.")
        
        logging.info(f"Using LM Studio model: {lm_studio_model} at {lm_studio_base_url}")
        return ChatOpenAI(
            api_key=lm_studio_api_key,
            base_url=f"{lm_studio_base_url}/v1",
            model=lm_studio_model,
            temperature=temperature
        )
    else:
        logging.error(f"Unsupported LLM provider: {provider}")
        raise ValueError(f"Unsupported LLM provider: {provider}")


def create_qa_chain(vector_db: VectorStore, llm: BaseLanguageModel) -> ConversationalRetrievalChain:
    try:
        retriever_k_str = os.getenv("RETRIEVER_K", "3")
        retriever_k = int(retriever_k_str)
        if retriever_k <= 0:
            logging.warning(f"RETRIEVER_K value ({retriever_k}) is invalid, defaulting to 3.")
            retriever_k = 3
    except ValueError:
        logging.warning(f"RETRIEVER_K value ('{retriever_k_str}') is not a valid integer, defaulting to 3.")
        retriever_k = 3

    logging.info(f"Using retriever_k = {retriever_k}")

    qa_chain: ConversationalRetrievalChain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": retriever_k}),
        return_source_documents=True,
    )
    return qa_chain

def answer_question(qa_chain: ConversationalRetrievalChain, question: str, chat_history: List[Tuple[str, str]]) -> Tuple[str, List[Document]]:
    result: Dict[str, Any] = qa_chain.invoke({
        "question": question,
        "chat_history": chat_history
    })
    return result["answer"], result["source_documents"]