import os
import logging # Import logging
from typing import Optional, Dict, Any, List, Tuple
from dotenv import load_dotenv
import shutil
from langchain_community.document_loaders import UnstructuredFileLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter # Changed import
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings # Or HuggingFaceEmbeddings
from langchain_community.embeddings import BedrockEmbeddings # Uncomment if using Bedrock
from langchain_openai import AzureOpenAIEmbeddings  # Correct import for Azure embeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document # For type hinting
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.language_models.llms import BaseLanguageModel # Import BaseLanguageModel
from langchain_core.embeddings import Embeddings # Import Embeddings
from langchain_core.vectorstores import VectorStore # Import VectorStore
from langchain_community.llms import Bedrock
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAI
from langchain_community.llms import Ollama
from langchain_openai import AzureOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
CHROMA_DB_DIRECTORY = "chroma_db_store" # Persistent storage directory
DATA_PATH = "../data" # Relative path to the data directory

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
        ollama_embedding_model: Optional[str] = os.getenv("OLLAMA_EMBEDDING_MODEL") # Use specific embedding model variable
        if not ollama_embedding_model:
            logging.error("OLLAMA_EMBEDDING_MODEL not found in environment variables for Ollama embedding provider.") 
            raise ValueError("OLLAMA_EMBEDDING_MODEL not found in environment variables for Ollama embedding provider.")
        # Ensure ollama server is running if using this
        logging.info(f"Using Ollama embedding model: {ollama_embedding_model} at {ollama_base_url}")
        return OllamaEmbeddings(model=ollama_embedding_model, base_url=ollama_base_url)
    
    elif provider == "azure":
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") # Use ENDPOINT consistently
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME") # Specific for embeddings
        if not all([azure_api_key, azure_endpoint, azure_api_version, azure_embedding_deployment]):
            logging.error("Missing required Azure OpenAI configuration (Key, Endpoint, Version, Embedding Deployment Name) in environment variables.")
            raise ValueError("Missing required Azure OpenAI configuration (Key, Endpoint, Version, Embedding Deployment Name) in environment variables.")
        logging.info(f"Using Azure embedding deployment: {azure_embedding_deployment}")
        return AzureOpenAIEmbeddings(
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            api_version=azure_api_version,
            azure_deployment=azure_embedding_deployment # Use specific deployment
        )
    
    elif provider == "bedrock":
        # Ensure necessary AWS credentials and region are configured in the environment
        # (e.g., AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
        # or via an IAM role or profile.
        model_id: Optional[str] = os.getenv("BEDROCK_EMBEDDING_MODEL_ID")
        region_name: Optional[str] = os.getenv("AWS_REGION") # Bedrock often requires region
        credentials_profile_name: Optional[str] = os.getenv("BEDROCK_PROFILE_NAME") # Optional: for specific AWS profiles

        if not model_id:
            logging.error("BEDROCK_EMBEDDING_MODEL_ID not found in environment variables for Bedrock embedding provider.")
            raise ValueError("BEDROCK_EMBEDDING_MODEL_ID not found in environment variables for Bedrock embedding provider.")
        if not region_name:
             logging.warning("AWS_REGION not set for Bedrock embeddings, it might default or fail.")

        logging.info(f"Using Bedrock embedding model: {model_id} in region {region_name or 'default'}")
        bedrock_params: Dict[str, Any] = {
            "model_id": model_id,
            # client=None, # Let LangChain handle client creation by default
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
def load_documents_from_directory(directory_path: str) -> List[Document]:
    """Loads documents from various file types in a directory."""
    documents: List[Document] = []
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
                # Load the entire JSON content as text for broader compatibility.
                # This might include JSON keys/structure in the loaded document.
                # For specific structures, you might revert to a targeted jq_schema.
                loader = JSONLoader(file_path=file_path, jq_schema='.', text_content=True)
                # Or if the whole object is content: loader = JSONLoader(file_path=file_path, jq_schema='.', text_content=True)
                loaded_json_docs = loader.load()
                # Add metadata if desired (e.g., source from the JSON)
                # For simplicity here, we just load the content.
                documents.extend(loaded_json_docs)
            elif filename.endswith(".csv"):
                loader = CSVLoader(file_path=file_path)
                documents.extend(loader.load())

            logging.debug(f"Loaded documents from: {filename}") # Use debug for per-file success
        except Exception as e:
            logging.error(f"Error loading {filename}: {e}", exc_info=True) # Log exception info

    logging.info(f"Finished loading documents. Total loaded: {len(documents)}")
    return documents

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]: # Added type hints
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
    vector_db: Optional[VectorStore] = None # Initialize vector_db

    if force_reload and documents:
        logging.info("Forcing reload of vector database...")
        if os.path.exists(persist_directory):
            logging.info(f"Removing existing database at {persist_directory}")
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

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        return OpenAI(api_key=api_key)
    elif provider == "ollama":
        ollama_base_url: Optional[str] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_llm_model: Optional[str] = os.getenv("OLLAMA_LLM_MODEL") # Use specific LLM model variable
        if not ollama_llm_model:
            logging.error("OLLAMA_LLM_MODEL not found in environment variables for Ollama LLM provider.")
            raise ValueError("OLLAMA_LLM_MODEL not found in environment variables for Ollama LLM provider.")
        logging.info(f"Using Ollama LLM model: {ollama_llm_model} at {ollama_base_url}")
        return Ollama(model=ollama_llm_model, base_url=ollama_base_url)
    # Add other providers:
    elif provider == "azure":
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") # Use ENDPOINT consistently
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        azure_llm_deployment = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT_NAME") # Specific for LLM
        if not all([azure_api_key, azure_endpoint, azure_api_version, azure_llm_deployment]):
            logging.error("Missing required Azure OpenAI configuration (Key, Endpoint, Version, LLM Deployment Name) in environment variables.")
            raise ValueError("Missing required Azure OpenAI configuration (Key, Endpoint, Version, LLM Deployment Name) in environment variables.")
        logging.info(f"Using Azure LLM deployment: {azure_llm_deployment}")
        return AzureOpenAI(
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint, 
            api_version=azure_api_version,
            azure_deployment=azure_llm_deployment # Use specific deployment
        )
    elif provider == "bedrock":
        
        # Ensure necessary AWS credentials and region are configured in the environment
        # (e.g., AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
        # or via an IAM role or profile.
        model_id = os.getenv("BEDROCK_MODEL_ID")
        region_name = os.getenv("AWS_REGION") # Bedrock often requires region
        credentials_profile_name = os.getenv("BEDROCK_PROFILE_NAME") # Optional: for specific AWS profiles

        if not model_id:
            logging.error("BEDROCK_MODEL_ID not found in environment variables for Bedrock provider.")
            raise ValueError("BEDROCK_MODEL_ID not found in environment variables for Bedrock provider.")
        if not region_name:
             logging.warning("AWS_REGION not set, Bedrock might default or fail.")

        logging.info(f"Using Bedrock model: {model_id} in region {region_name or 'default'}")
        bedrock_params: Dict[str, Any] = {
            "model_id": model_id,
            # client=None, # Let LangChain handle client creation by default
        }
        if region_name:
            bedrock_params["region_name"] = region_name
        if credentials_profile_name:
            bedrock_params["credentials_profile_name"] = credentials_profile_name

        # Add model_kwargs if needed, e.g., for temperature:
        # bedrock_params["model_kwargs"] = {"temperature": 0.7}

        return Bedrock(**bedrock_params)
    else:
        logging.error(f"Unsupported LLM provider: {provider}")
        raise ValueError(f"Unsupported LLM provider: {provider}")

# --- QA Chain ---
def create_qa_chain(vector_db: VectorStore, llm: BaseLanguageModel) -> RetrievalQA:
    """Creates the RetrievalQA chain with a specific prompt."""
    # No null check needed due to type hint
    # if not vector_db:
    #      raise ValueError("Vector database is not initialized.")

    template = """
        Answer the following question based only on the provided context.
        If the context does not contain the answer, state that you cannot answer based on the provided context.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    
    # Adjust the prompt template as needed
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain: RetrievalQA = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # Loads all relevant chunks into the context window
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}), # Retrieve top 3 relevant chunks
        return_source_documents=True, # Return the source chunks
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain

def answer_question(qa_chain: RetrievalQA, question: str) -> Tuple[str, List[Document]]:
    """Answers a question using the QA chain."""
    # No null check needed due to type hint
    # if not qa_chain:
    #      raise ValueError("QA chain is not initialized.")
    result: Dict[str, Any] = qa_chain.invoke({"query": question}) # Use invoke for newer Langchain versions
    return result["result"], result["source_documents"]