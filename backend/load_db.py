import os
from utils import (
    load_documents_from_directory,
    create_or_load_vector_db,
    get_embedding_function,
    DATA_PATH
)

if __name__ == "__main__":
    print("--- Starting Data Loading and Vector DB Creation ---")

    # Ensure the data path exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data directory not found at {DATA_PATH}")
        exit(1)

    # 1. Load documents from the specified directory
    print(f"Loading documents from: {DATA_PATH}")
    documents = load_documents_from_directory(DATA_PATH)
    if not documents:
        print("No documents loaded. Exiting.")
        exit(1)
    print(f"Total documents loaded: {len(documents)}")

    # 2. Get the embedding function (based on .env settings)
    try:
        embedding_func = get_embedding_function()
    except ValueError as e:
        print(f"Error getting embedding function: {e}")
        exit(1)

    # 3. Create/rebuild the vector database (force_reload=True ensures fresh build)
    try:
        vector_db = create_or_load_vector_db(
            documents=documents,
            embedding_function=embedding_func,
            force_reload=True # Set to True to always rebuild when this script runs
        )
        if vector_db:
            print("--- Vector DB Creation/Update Complete ---")
        else:
            print("--- Vector DB Creation Failed ---")

    except Exception as e:
        print(f"An error occurred during vector DB creation: {e}")
        exit(1)