"""
RAG Retriever - handles document retrieval and context building
"""
import yaml
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict

class RAGRetriever:
    """Retrieve relevant documents for queries"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize embeddings
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config['embeddings']['model_name'],
            model_kwargs={'device': self.config['embeddings']['device']}
        )
        
        # Load vector store
        print("Loading vector store...")
        self.vector_store = Chroma(
            persist_directory=self.config['vector_store']['persist_directory'],
            embedding_function=self.embeddings,
            collection_name=self.config['vector_store']['collection_name']
        )
        
        print("✓ RAG Retriever initialized")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve relevant documents for a query"""
        if top_k is None:
            top_k = self.config['rag']['top_k']
        
        # Search for similar documents
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        
        # Format results
        retrieved_docs = []
        for doc, score in results:
            retrieved_docs.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'relevance_score': float(score)
            })
        
        return retrieved_docs
    
    def build_context(self, query: str) -> str:
        """Build context string from retrieved documents"""
        docs = self.retrieve(query)
        
        if not docs:
            return "No relevant information found."
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"Document {i}:\n{doc['content']}")
        
        return "\n\n".join(context_parts)

if __name__ == "__main__":
    # Test the retriever
    retriever = RAGRetriever()
    
    test_queries = [
        "What is machine learning?",
        "Tell me about MLOps",
        "What are Python libraries for data science?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        context = retriever.build_context(query)
        print(f"\nRetrieved Context:\n{context[:500]}...")
