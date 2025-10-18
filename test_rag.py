"""
Test the RAG system
"""
from src.rag.pipeline import RAGPipeline

def main():
    print("Initializing RAG Pipeline...")
    pipeline = RAGPipeline()
    
    test_questions = [
        "What is machine learning?",
        "Explain MLOps and its key components",
        "What Python libraries are essential for data science?",
        "How does supervised learning differ from unsupervised learning?",
        "What tools are used for experiment tracking in MLOps?"
    ]
    
    for question in test_questions:
        print(f"\n{'='*70}")
        print(f"❓ Question: {question}")
        print('='*70)
        
        result = pipeline.query(question, return_sources=True)
        
        print(f"\n💡 Answer:\n{result['answer']}")
        print(f"\n📚 Sources used: {result['num_sources']}")
        
        if result.get('sources'):
            print("\n📄 Source excerpts:")
            for i, source in enumerate(result['sources'][:2], 1):
                print(f"\n  {i}. Score: {source['relevance_score']:.3f}")
                print(f"     {source['content'][:150]}...")

if __name__ == "__main__":
    main()
