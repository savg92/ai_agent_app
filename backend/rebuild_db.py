#!/usr/bin/env python3
"""
Utility script to rebuild the vector database.
Useful when changing embedding providers or models.
"""

import argparse
import logging
import os
import sys
from dotenv import load_dotenv

# Add the backend directory to the path so we can import utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    load_documents_from_directory,
    get_embedding_function,
    create_or_load_vector_db,
    DATA_PATH,
    CHROMA_DB_DIRECTORY
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description='Rebuild the vector database')
    parser.add_argument('--force', action='store_true', 
                       help='Force rebuild even if database exists')
    parser.add_argument('--data-path', type=str, default=DATA_PATH,
                       help=f'Path to data directory (default: {DATA_PATH})')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    logging.info("Starting vector database rebuild...")
    
    try:
        # Load documents
        logging.info(f"Loading documents from: {args.data_path}")
        documents, failed_files = load_documents_from_directory(args.data_path)
        
        if not documents:
            logging.error("No documents loaded. Cannot rebuild database.")
            return 1
        
        if failed_files:
            logging.warning(f"Some files failed to load: {failed_files}")
        
        # Get embedding function
        embedding_func = get_embedding_function()
        
        # Rebuild database
        vector_db = create_or_load_vector_db(
            documents=documents,
            embedding_function=embedding_func,
            force_reload=args.force or True  # Always force rebuild in this script
        )
        
        if vector_db is None:
            logging.error("Failed to create vector database")
            return 1
        
        logging.info(f"Vector database successfully rebuilt at: {CHROMA_DB_DIRECTORY}")
        return 0
        
    except Exception as e:
        logging.error(f"Error rebuilding database: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
