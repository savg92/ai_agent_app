"""Vector store service for managing embeddings and retrieval."""

import logging
from typing import Optional, List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_chroma import Chroma

from services.document_service import DocumentService


class VectorStoreService:
    """Service for managing vector stores."""
    
    def __init__(self, document_service: DocumentService):
        self.document_service = document_service
    
    def create_vector_store(
        self, 
        documents: List[Document], 
        embedding_function: Embeddings, 
        persist_directory: str
    ) -> VectorStore:
        """Create a new vector store from documents."""
        logging.info("Splitting documents...")
        texts = self.document_service.split_documents(documents)
        
        if not texts:
            logging.warning("No text chunks to add to the database after splitting.")
            raise ValueError("No text chunks available after document splitting.")
        
        logging.info(f"Creating new vector database with {len(texts)} chunks...")
        vector_db = Chroma.from_documents(
            documents=texts,
            embedding=embedding_function,
            persist_directory=persist_directory
        )
        logging.info(f"Vector database created and persisted at {persist_directory}")
        return vector_db
    
    def load_vector_store(
        self, 
        embedding_function: Embeddings, 
        persist_directory: str
    ) -> VectorStore:
        """Load an existing vector store."""
        logging.info(f"Loading existing vector database from {persist_directory}...")
        vector_db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )
        logging.info("Vector database loaded.")
        return vector_db
