from langchain_deepseek import ChatDeepSeek
from file_processor import FileProcessor
from typing import Optional, Dict, Any
import logging

class RAGSystem:
    def __init__(self):
        self.model = ChatDeepSeek(
            model_name="deepseek/chat-7b",
            temperature=0.1,
            max_tokens=1024,
            top_p=0.95,
            top_k=40,
        )
        self.file_processor = FileProcessor()
        self.vector_store = self.file_processor.get_vector_store()
        self.logger = logging.getLogger(__name__)

    def retrieve(self, query: str) -> Optional[list]:
        """Retrieve relevant documents from the vector store."""
        try:
            return self.vector_store.similarity_search(query, k=3)
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {e}")
            return None

    def generate_response(self, query: str) -> Dict[str, Any]:
        """Generate a response based on the retrieved documents."""
        try:
            documents = self.retrieve(query)
            if not documents:
                return {"error": "Failed to retrieve documents"}
            
            # Format documents for the model
            context = "\n\n".join([doc.page_content for doc in documents])
            return self.model.generate(
                prompt=query,
                context=context
            )
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {"error": str(e)}

    def run(self, query: str) -> Dict[str, Any]:
        """Run the RAG system to get a response for the given query."""
        try:
            response = self.generate_response(query)
            if "error" in response:
                self.logger.warning(f"RAG system failed for query: {query}")
            return response
        except Exception as e:
            self.logger.error(f"Unexpected error in RAG system: {e}")
            return {"error": "Internal server error"}