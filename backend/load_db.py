import os
import argparse
import logging
from utils import (
    load_documents_from_directory,
    create_or_load_vector_db,
    get_embedding_function,
    DATA_PATH
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load data and build/update the vector database.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Force rebuild of the vector database even if it exists."
    )
    args = parser.parse_args()
    force_reload_db = args.reload

    logging.info("--- Starting Data Loading and Vector DB Creation ---")
    if force_reload_db:
        logging.info("*** Force Reload Requested ***")

    if not os.path.exists(DATA_PATH):
        logging.error(f"Error: Data directory not found at {DATA_PATH}")
        exit(1)

    logging.info(f"Loading documents from: {DATA_PATH}")
    documents, failed_files = load_documents_from_directory(DATA_PATH)

    if failed_files:
        logging.warning(f"The following files failed to load and were skipped: {failed_files}")

    if not documents:
        logging.error("No documents loaded successfully. Exiting.")
        exit(1)
    logging.info(f"Total documents loaded successfully: {len(documents)}")

    try:
        embedding_func = get_embedding_function()
    except ValueError as e:
        logging.error(f"Error getting embedding function: {e}")
        exit(1)

    try:
        vector_db = create_or_load_vector_db(
            documents=documents,
            embedding_function=embedding_func,
            force_reload=force_reload_db
        )
        if vector_db:
            logging.info("--- Vector DB Creation/Update Complete ---")
        else:
            logging.error("--- Vector DB Creation Failed (possibly no text chunks after splitting) ---")
            exit(1)

    except Exception as e:
        logging.error(f"An error occurred during vector DB creation: {e}", exc_info=True)
        exit(1)