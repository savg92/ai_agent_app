"""Document loading and processing service."""

import os
import logging
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredFileLoader, JSONLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentService:
    """Service for loading and processing documents."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\f", "\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    
    def load_documents_from_directory(self, directory_path: str) -> Tuple[List[Document], List[str]]:
        """Load documents from a directory."""
        documents: List[Document] = []
        failed_files: List[str] = []
        
        logging.info(f"Loading documents from directory: {directory_path}")
        
        if not os.path.exists(directory_path):
            logging.error(f"Directory does not exist: {directory_path}")
            return documents, failed_files
        
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
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        logging.info(f"Splitting {len(documents)} documents using separators: {self.text_splitter._separators}")
        split_docs: List[Document] = self.text_splitter.split_documents(documents)
        logging.info(f"Split into {len(split_docs)} chunks.")
        return split_docs
