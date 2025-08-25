"""QA service for handling question answering logic."""

import logging
from typing import List, Tuple, Dict, Any
from langchain.chains import ConversationalRetrievalChain
from langchain_core.language_models.base import BaseLanguageModel
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
        search_type = getattr(self.settings, 'retriever_search_type', 'similarity')
        score_threshold = getattr(self.settings, 'retriever_score_threshold', None)
        logging.info(f"Retriever config: k={retriever_k}, type={search_type}, threshold={score_threshold}")

        search_kwargs: Dict[str, Any] = {"k": retriever_k}
        if search_type == "mmr":
            search_kwargs.update({"fetch_k": max(retriever_k * 2, retriever_k + 1), "lambda_mult": 0.5})
        if search_type == "similarity_score_threshold" and score_threshold is not None:
            search_kwargs.update({"score_threshold": score_threshold})

        qa_chain: ConversationalRetrievalChain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_db.as_retriever(search_type=search_type, search_kwargs=search_kwargs),
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
        # Trim chat history to reduce tokens
        max_turns = getattr(self.settings, 'chat_history_max_turns', 8)
        trimmed_history = chat_history[-max_turns:] if max_turns and max_turns > 0 else []

        result: Dict[str, Any] = qa_chain.invoke({
            "question": question,
            "chat_history": trimmed_history
        })
        return result["answer"], result["source_documents"]
