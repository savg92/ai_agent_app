"""Path configuration for the application."""

import pathlib
from typing import Optional


class Paths:
    """Application paths configuration."""
    
    def __init__(self):
        self.backend_dir = pathlib.Path(__file__).parent.parent.resolve()
        self.project_root = self.backend_dir.parent.resolve()
        
        # Data and storage paths
        self.data_path = str(self.project_root / "data")
        self.vector_stores_dir = self.backend_dir / "vector_stores"
        self.chroma_db_directory = str(self.backend_dir / "chroma_db_store")  # Legacy support
        
        # Configuration files
        self.embedding_config_file = str(self.backend_dir / "embedding_config.json")
        self.active_store_config = self.backend_dir / "active_vector_store.json"
        
        # Ensure directories exist
        self.vector_stores_dir.mkdir(exist_ok=True)


# Global paths instance
_paths: Optional[Paths] = None


def get_paths() -> Paths:
    """Get the global paths instance."""
    global _paths
    if _paths is None:
        _paths = Paths()
    return _paths
