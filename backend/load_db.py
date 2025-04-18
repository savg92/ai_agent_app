import os
import argparse
import logging
from utils import (
    load_documents_from_directory,
    create_or_load_vector_db,
    get_embedding_function,
    DATA_PATH
)

# Configure logging (optional, if not configured elsewhere)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Load data and build/update the vector database.")
    parser.add_argument(
        "--reload",
        action="store_true", # Sets reload_db to True if flag is present
        help="Force rebuild of the vector database even if it exists."
    )
    args = parser.parse_args()
    force_reload_db = args.reload

    logging.info("--- Starting Data Loading and Vector DB Creation ---")
    if force_reload_db:
        logging.info("*** Force Reload Requested ***")

    # Ensure the data path exists
    if not os.path.exists(DATA_PATH):
        logging.error(f"Error: Data directory not found at {DATA_PATH}")
        exit(1)

    # 1. Load documents from the specified directory
    logging.info(f"Loading documents from: {DATA_PATH}")
    # Unpack the returned tuple
    documents, failed_files = load_documents_from_directory(DATA_PATH)

    # Log failed files if any
    if failed_files:
        logging.warning(f"The following files failed to load and were skipped: {failed_files}")

    # Exit if NO documents were loaded successfully
    if not documents:
        logging.error("No documents loaded successfully. Exiting.") 
        exit(1)
    logging.info(f"Total documents loaded successfully: {len(documents)}") 

    # 2. Get the embedding function (based on .env settings)
    try:
        embedding_func = get_embedding_function()
    except ValueError as e:
        logging.error(f"Error getting embedding function: {e}") 
        exit(1)

    # 3. Create/rebuild the vector database
    try:
        vector_db = create_or_load_vector_db(
            documents=documents,
            embedding_function=embedding_func,
            force_reload=force_reload_db
        )
        if vector_db:
            logging.info("--- Vector DB Creation/Update Complete ---")
        else:
            # This case might happen if splitting results in no texts, handled in create_or_load_vector_db
            logging.error("--- Vector DB Creation Failed (possibly no text chunks after splitting) ---")
            exit(1) # Exit if DB creation failed

    except Exception as e:
        logging.error(f"An error occurred during vector DB creation: {e}", exc_info=True)
        exit(1) # Exit if a critical error occurs during DB creation