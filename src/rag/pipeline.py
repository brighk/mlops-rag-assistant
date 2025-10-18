"""
RAG Pipeline - combines retrieval and generation
"""
import yaml
from src.rag.retriever import RAGRetriever
from src.models.llm_ollama import OllamaLLM  # Using Ollama now!
import mlflow
from datetime import datetime

class RAGPipeline:
    """Complete RAG pipeline with Ollama"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print("Initializing RAG Pipeline...")
        
        # Initialize components
        self.retriever = RAGRetriever(config_path)
        self.llm = OllamaLLM(config_path)
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
        print("✓ RAG Pipeline ready!")
    
    def query(self, question: str, return_sources: bool = True) -> dict:
        """Query the RAG system"""
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log query
            mlflow.log_param("question", question)
            
            # 1. Retrieve relevant documents
            print("\n🔍 Retrieving relevant documents...")
            retrieved_docs = self.retriever.retrieve(question)
            context = self.retriever.build_context(question)
            
            mlflow.log_metric("num_retrieved_docs", len(retrieved_docs))
            
            # 2. Format prompt
            prompt = self.llm.format_prompt(question, context)
            
            # 3. Generate answer
            print("🤖 Generating answer...")
            answer = self.llm.generate(prompt, max_length=300)
            
            # Log to MLflow
            mlflow.log_text(answer, "answer.txt")
            mlflow.log_text(context, "context.txt")
            
            result = {
                "question": question,
                "answer": answer,
                "context": context if return_sources else None,
                "sources": retrieved_docs if return_sources else None,
                "num_sources": len(retrieved_docs)
            }
            
            return result

if __name__ == "__main__":
    # Test the complete pipeline
    pipeline = RAGPipeline()
    
    test_questions = [
        "What is machine learning?",
        "What are the key components of MLOps?",
        "Which Python libraries should I use for data science?"
    ]
    
    for question in test_questions:
        print(f"\n{'='*70}")
        print(f"❓ Question: {question}")
        print('='*70)
        
        result = pipeline.query(question, return_sources=False)
        
        print(f"\n💡 Answer:\n{result['answer']}")
        print(f"\n📚 Used {result['num_sources']} source documents")
