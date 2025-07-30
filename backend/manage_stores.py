#!/usr/bin/env python3
"""
Vector Store Management CLI

This script provides command-line utilities for managing multiple vector stores
in the RAG application. You can list, delete, switch between, and rebuild
vector stores for different embedding configurations.

Usage:
    python manage_stores.py list              # List all vector stores
    python manage_stores.py delete <id>       # Delete a specific store
    python manage_stores.py delete-all        # Delete all stores
    python manage_stores.py rebuild           # Rebuild current store
    python manage_stores.py current           # Show current active config
"""

import argparse
import sys
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Add the backend directory to path so we can import utils
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    VectorStoreManager,
    get_current_embedding_config,
    get_embedding_function,
    load_documents_from_directory,
    DATA_PATH
)

def setup_environment():
    """Load environment variables."""
    load_dotenv()

def format_store_info(store):
    """Format store information for display."""
    lines = [
        f"  Identifier: {store.get('identifier', 'Unknown')}",
        f"  Provider: {store.get('provider', 'Unknown')}",
        f"  Model: {store.get('model', store.get('deployment', store.get('model_id', 'Unknown')))}",
        f"  Created: {store.get('created_at', 'Unknown')}",
        f"  Last Used: {store.get('last_used', 'Unknown')}",
        f"  Documents: {store.get('document_count', 0)}",
        f"  Chunks: {store.get('chunk_count', 0)}",
    ]
    return "\n".join(lines)

def list_stores():
    """List all available vector stores."""
    manager = VectorStoreManager()
    stores = manager.list_available_stores()
    active_config = manager.get_active_store_config()
    
    print(f"Found {len(stores)} vector store(s):")
    print("=" * 50)
    
    if not stores:
        print("No vector stores found.")
        return
    
    for i, store in enumerate(stores, 1):
        is_active = (active_config and 
                    store.get('identifier') == manager.get_vector_store_identifier(active_config))
        status = " (ACTIVE)" if is_active else ""
        
        print(f"{i}. {store.get('identifier', 'Unknown')}{status}")
        print(format_store_info(store))
        print("-" * 30)

def delete_store(identifier):
    """Delete a specific vector store."""
    manager = VectorStoreManager()
    
    # Check if store exists
    stores = manager.list_available_stores()
    if not any(store.get('identifier') == identifier for store in stores):
        print(f"Error: Vector store '{identifier}' not found.")
        return False
    
    success = manager.delete_store(identifier)
    if success:
        print(f"Vector store '{identifier}' deleted successfully.")
        return True
    else:
        print(f"Failed to delete vector store '{identifier}'.")
        return False

def delete_all_stores():
    """Delete all vector stores."""
    manager = VectorStoreManager()
    stores = manager.list_available_stores()
    
    if not stores:
        print("No vector stores to delete.")
        return
    
    print(f"This will delete {len(stores)} vector store(s):")
    for store in stores:
        print(f"  - {store.get('identifier')}")
    
    confirm = input("Are you sure? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Operation cancelled.")
        return
    
    manager.delete_all_stores()
    print("All vector stores deleted successfully.")

def show_current_config():
    """Show the current active embedding configuration."""
    manager = VectorStoreManager()
    current_config = get_current_embedding_config()
    active_config = manager.get_active_store_config()
    
    print("Current Embedding Configuration:")
    print("=" * 40)
    print(json.dumps(current_config, indent=2))
    
    print("\nActive Store Configuration:")
    print("=" * 40)
    if active_config:
        print(json.dumps(active_config, indent=2))
    else:
        print("No active store configuration found.")
    
    # Check if current config matches any existing store
    stores = manager.list_available_stores()
    current_id = manager.get_vector_store_identifier(current_config)
    matching_store = next((s for s in stores if s.get('identifier') == current_id), None)
    
    print(f"\nMatching Store: {'Found' if matching_store else 'Not Found'}")
    if matching_store:
        print(f"Store ID: {current_id}")

def rebuild_current_store():
    """Rebuild the current vector store."""
    try:
        manager = VectorStoreManager()
        current_config = get_current_embedding_config()
        
        print("Current configuration:")
        print(json.dumps(current_config, indent=2))
        print()
        
        # Check if data directory exists
        if not os.path.exists(DATA_PATH):
            print(f"Error: Data directory not found at {DATA_PATH}")
            return False
        
        # Load documents
        print(f"Loading documents from {DATA_PATH}...")
        documents, failed_files = load_documents_from_directory(DATA_PATH)
        
        if failed_files:
            print(f"Warning: {len(failed_files)} files failed to load:")
            for file in failed_files:
                print(f"  - {file}")
        
        if not documents:
            print("Error: No documents loaded successfully.")
            return False
        
        print(f"Loaded {len(documents)} documents.")
        
        # Ask for confirmation
        identifier = manager.get_vector_store_identifier(current_config)
        stores = manager.list_available_stores()
        existing_store = next((s for s in stores if s.get('identifier') == identifier), None)
        
        if existing_store:
            print(f"This will replace the existing store: {identifier}")
            confirm = input("Continue? (y/N): ").strip().lower()
            if confirm != 'y':
                print("Operation cancelled.")
                return False
            # Delete existing store
            manager.delete_store(identifier)
        
        # Create new store
        print("Creating vector store...")
        embedding_func = get_embedding_function()
        vector_db = manager.get_or_create_store(
            config=current_config,
            documents=documents,
            embedding_function=embedding_func
        )
        
        manager.set_active_store(current_config)
        
        print(f"Vector store rebuilt successfully!")
        print(f"Store ID: {identifier}")
        print(f"Documents: {len(documents)}")
        
        return True
        
    except Exception as e:
        print(f"Error rebuilding vector store: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Manage vector stores for the RAG application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    subparsers.add_parser('list', help='List all vector stores')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a specific vector store')
    delete_parser.add_argument('identifier', help='Store identifier to delete')
    
    # Delete all command
    subparsers.add_parser('delete-all', help='Delete all vector stores')
    
    # Current command
    subparsers.add_parser('current', help='Show current embedding configuration')
    
    # Rebuild command
    subparsers.add_parser('rebuild', help='Rebuild current vector store')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    setup_environment()
    
    try:
        if args.command == 'list':
            list_stores()
        elif args.command == 'delete':
            delete_store(args.identifier)
        elif args.command == 'delete-all':
            delete_all_stores()
        elif args.command == 'current':
            show_current_config()
        elif args.command == 'rebuild':
            rebuild_current_store()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
