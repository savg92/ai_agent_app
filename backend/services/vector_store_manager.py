"""Vector store manager for handling multiple vector stores."""

import json
import logging
import shutil
import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from config import get_paths
from core.types import EmbeddingConfig
from services.vector_store_service import VectorStoreService


class VectorStoreManager:
    """Manager for handling multiple vector stores."""
    
    def __init__(self, vector_store_service: VectorStoreService):
        self.paths = get_paths()
        self.vector_store_service = vector_store_service
        self.stores_dir = self.paths.vector_stores_dir
        self.active_config_file = self.paths.active_store_config
        self.stores_dir.mkdir(exist_ok=True)
    
    def get_vector_store_identifier(self, config: EmbeddingConfig) -> str:
        """Generate a unique identifier for a vector store configuration."""
        provider = config.get('provider', 'unknown')
        model = (config.get('model') or 
                config.get('deployment') or 
                config.get('model_id') or 'default')
        # Normalize for filesystem
        safe_model = str(model).replace(':', '_').replace('/', '_').replace(' ', '_')
        return f"{provider}_{safe_model}"
    
    def get_vector_store_path(self, config: EmbeddingConfig) -> Path:
        """Get the path for a vector store."""
        return self.stores_dir / self.get_vector_store_identifier(config)
    
    def save_store_metadata(
        self, 
        store_path: Path, 
        config: EmbeddingConfig, 
        doc_count: int, 
        chunk_count: int
    ) -> None:
        """Save metadata for a vector store."""
        meta = dict(config)
        meta['created_at'] = datetime.datetime.utcnow().isoformat() + 'Z'
        meta['document_count'] = doc_count
        meta['chunk_count'] = chunk_count
        meta['last_used'] = meta['created_at']
        meta['version'] = '1.0'
        
        with open(store_path / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)
    
    def load_store_metadata(self, store_path: Path) -> Dict[str, Any]:
        """Load metadata for a vector store."""
        try:
            with open(store_path / 'metadata.json') as f:
                return json.load(f)
        except Exception:
            return {}
    
    def list_available_stores(self) -> List[Dict[str, Any]]:
        """List all available vector stores."""
        stores = []
        for store_dir in self.stores_dir.iterdir():
            if store_dir.is_dir():
                meta = self.load_store_metadata(store_dir)
                meta['identifier'] = store_dir.name
                meta['path'] = str(store_dir)
                stores.append(meta)
        return stores
    
    def get_or_create_store(
        self, 
        config: EmbeddingConfig, 
        documents: Optional[List[Document]] = None, 
        embedding_function: Optional[Embeddings] = None
    ) -> VectorStore:
        """Get an existing vector store or create a new one."""
        store_path = self.get_vector_store_path(config)
        persist_directory = str(store_path)
        
        if store_path.exists() and (store_path / 'chroma.sqlite3').exists():
            # Load existing
            if embedding_function is None:
                raise ValueError("An Embeddings instance is required to load an existing vector store.")
            vector_db = self.vector_store_service.load_vector_store(
                embedding_function, persist_directory
            )
            # Update last_used
            meta = self.load_store_metadata(store_path)
            meta['last_used'] = datetime.datetime.utcnow().isoformat() + 'Z'
            with open(store_path / 'metadata.json', 'w') as f:
                json.dump(meta, f, indent=2)
            return vector_db
        else:
            # Create new
            if not documents:
                raise ValueError("No documents provided to create new vector store.")
            if embedding_function is None:
                raise ValueError("An Embeddings instance is required to create a new vector store.")
            
            store_path.mkdir(exist_ok=True)
            vector_db = self.vector_store_service.create_vector_store(
                documents, embedding_function, persist_directory
            )
            
            # Calculate chunk count
            chunk_count = len(self.vector_store_service.document_service.split_documents(documents))
            
            self.save_store_metadata(store_path, config, len(documents), chunk_count)
            return vector_db
    
    def delete_store(self, identifier: str) -> bool:
        """Delete a vector store."""
        store_path = self.stores_dir / identifier
        if store_path.exists():
            shutil.rmtree(store_path)
            return True
        return False
    
    def delete_all_stores(self) -> None:
        """Delete all vector stores."""
        for store_dir in self.stores_dir.iterdir():
            if store_dir.is_dir():
                shutil.rmtree(store_dir)
    
    def set_active_store(self, config: EmbeddingConfig) -> None:
        """Set the active vector store configuration."""
        with open(self.active_config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def get_active_store_config(self) -> Dict[str, Any]:
        """Get the active vector store configuration."""
        if self.active_config_file.exists():
            with open(self.active_config_file) as f:
                return json.load(f)
        return {}
