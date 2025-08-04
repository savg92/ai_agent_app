"""QA service for handling question answering logic."""

import logging
from typing import List, Tuple, Dict, Any
from langchain.chains import ConversationalRetrievalChain
from langchain_core.language_models.llms import BaseLanguageModel
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document

from config import get_settings


class QAService:
    """Service for question answering operations."""
    
    def __init__(self):
        self.settings = get_settings()
    
    def create_qa_chain(
        self, 
        vector_db: VectorStore, 
        llm: BaseLanguageModel
    ) -> ConversationalRetrievalChain:
        """Create a conversational retrieval chain."""
        retriever_k = self.settings.retriever_k
        logging.info(f"Using retriever_k = {retriever_k}")

        qa_chain: ConversationalRetrievalChain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_db.as_retriever(search_kwargs={"k": retriever_k}),
            return_source_documents=True,
        )
        return qa_chain
    
    def answer_question(
        self, 
        qa_chain: ConversationalRetrievalChain, 
        question: str, 
        chat_history: List[Tuple[str, str]]
    ) -> Tuple[str, List[Document]]:
        """Answer a question using the QA chain."""
        result: Dict[str, Any] = qa_chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
        return result["answer"], result["source_documents"]
