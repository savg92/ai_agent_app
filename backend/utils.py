import os
import logging
from typing import Optional, Dict, Any, List, Tuple
from dotenv import load_dotenv
import shutil
from langchain_community.document_loaders import UnstructuredFileLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.language_models.llms import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.llms import Bedrock
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAI
from langchain_community.llms import Ollama
from langchain_openai import AzureOpenAI
import pathlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Get the directory where utils.py resides
BACKEND_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = BACKEND_DIR.parent.resolve()
# Define paths relative to the project structure
CHROMA_DB_DIRECTORY = str(BACKEND_DIR / "chroma_db_store")
DATA_PATH = str(PROJECT_ROOT / "data")

# --- Embedding Model Selection ---
def get_embedding_function(provider: str = os.getenv("EMBEDDING_PROVIDER", "openai")) -> Embeddings:
    """Selects and returns the embedding function based on the provider."""
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
        # Ensure ollama server is running if using this
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
            bedrock_params["credentials_profile_name"] = credentials_profile_name

        return BedrockEmbeddings(**bedrock_params)

    else:
        # Default or fallback: Using Sentence Transformers (works well locally)
        logging.info("Provider not explicitly supported or specified, defaulting to Sentence Transformers (all-MiniLM-L6-v2).")
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# --- Data Loading and Processing ---
def load_documents_from_directory(directory_path: str) -> Tuple[List[Document], List[str]]:
    """
    Loads documents from various file types in a directory.
    Returns a tuple containing a list of loaded documents and a list of failed file paths.
    """
    documents: List[Document] = []
    failed_files: List[str] = [] # Keep track of files that failed to load
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
                # Use text_content=False (default) to create a Document per JSON object (if list)
                # The page_content will be the string representation of the object.
                loader = JSONLoader(file_path=file_path, jq_schema='.', text_content=False)
                loaded_json_docs = loader.load()
                logging.info(f"Loaded {len(loaded_json_docs)} document(s) from JSON file: {filename}")
                documents.extend(loaded_json_docs)
            elif filename.endswith(".csv"):
                loader = CSVLoader(file_path=file_path)
                documents.extend(loader.load())

            logging.debug(f"Loaded documents from: {filename}")
        except Exception as e:
            # Log the error and add the file path to the failed list
            logging.error(f"Error loading {file_path}: {e}", exc_info=True)
            failed_files.append(file_path)

    logging.info(f"Finished loading documents. Total loaded: {len(documents)}. Failed files: {len(failed_files)}")
    if failed_files:
        logging.warning(f"Failed to load the following files: {failed_files}")
    return documents, failed_files

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
    """Splits documents into smaller chunks using multiple separators including page breaks."""
    # Using RecursiveCharacterTextSplitter for multiple separators.
    # It tries separators in order: page break, double newline, single newline, space, then characters.
    # Adjust '\f' if your documents use a different page break marker.
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\f", "\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    logging.info(f"Splitting {len(documents)} documents using separators: {text_splitter.separators}")
    split_docs: List[Document] = text_splitter.split_documents(documents)
    logging.info(f"Split into {len(split_docs)} chunks.")
    return split_docs

# --- Vector Database Operations ---
def create_or_load_vector_db(
    documents: Optional[List[Document]] = None,
    embedding_function: Optional[Embeddings] = None,
    force_reload: bool = False
) -> Optional[VectorStore]:
    """
    Creates a new Chroma vector database or loads an existing one.
    If documents are provided and force_reload is True, it rebuilds the DB.
    """
    if not embedding_function:
        embedding_function = get_embedding_function()

    persist_directory: str = CHROMA_DB_DIRECTORY
    vector_db: Optional[VectorStore] = None

    if force_reload and documents:
        logging.info("Forcing reload of vector database...")
        if os.path.exists(persist_directory):
            logging.warning(f"*** WARNING: Removing existing database directory due to force_reload=True: {persist_directory}")
            shutil.rmtree(persist_directory) # Be careful with this in production!

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
        return vector_db
    elif os.path.exists(persist_directory):
        logging.info(f"Loading existing vector database from {persist_directory}...")
        vector_db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )
        logging.info("Vector database loaded.")
        return vector_db
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
         return vector_db
    else:
        logging.error("Cannot create or load vector DB: No documents provided and no existing database found.")
        return None


# --- LLM Selection ---
def get_llm(provider: str = os.getenv("LLM_PROVIDER", "openai")) -> BaseLanguageModel:
    """Selects and initializes the LLM based on the provider."""
    provider = provider.lower()
    logging.info(f"Using LLM provider: {provider}")

    # Read and validate temperature from environment variable
    default_temp = 0.7
    try:
        temp_str = os.getenv("LLM_TEMPERATURE")
        if temp_str is None:
            temperature = default_temp
            logging.info(f"LLM_TEMPERATURE not set, using default: {temperature}")
        else:
            temperature = float(temp_str)
            # Basic range check, adjust as needed
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
        return Ollama(model=ollama_llm_model, base_url=ollama_base_url, temperature=temperature)
    elif provider == "azure":
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        azure_llm_deployment = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT_NAME")
        if not all([azure_api_key, azure_endpoint, azure_api_version, azure_llm_deployment]):
            logging.error("Missing required Azure OpenAI configuration (Key, Endpoint, Version, LLM Deployment Name) in environment variables.")
            raise ValueError("Missing required Azure OpenAI configuration (Key, Endpoint, Version, LLM Deployment Name) in environment variables.")
        logging.info(f"Using Azure LLM deployment: {azure_llm_deployment}")
        return AzureOpenAI(
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
            bedrock_params["credentials_profile_name"] = credentials_profile_name

        # Pass temperature via model_kwargs for Bedrock
        bedrock_params["model_kwargs"] = {"temperature": temperature}

        return Bedrock(**bedrock_params)
    else:
        logging.error(f"Unsupported LLM provider: {provider}")
        raise ValueError(f"Unsupported LLM provider: {provider}")

# --- QA Chain --- 
def create_qa_chain(vector_db: VectorStore, llm: BaseLanguageModel) -> RetrievalQA:
    """Creates the RetrievalQA chain with a specific prompt."""
    # Updated template to be more directive about using ONLY relevant context
    template = """Answer the question at the end based *only* on the following relevant context.
    If the provided context does not contain the answer to the question, just say that you don't know.
    Do not use any information from the context that is not directly related to the question asked.

    Use three sentences maximum and keep the answer as concise as possible.

    Relevant Context: {context}

    Question: {question}

    Helpful Answer:
    """

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Read k from environment variable, default to 3
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

    qa_chain: RetrievalQA = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # Loads all relevant chunks into the context window
        retriever=vector_db.as_retriever(search_kwargs={"k": retriever_k}), # Use configurable k
        return_source_documents=True, # Return the source chunks
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain

def answer_question(qa_chain: RetrievalQA, question: str) -> Tuple[str, List[Document]]:
    """Answers a question using the QA chain."""
    result: Dict[str, Any] = qa_chain.invoke({"query": question})
    return result["result"], result["source_documents"]