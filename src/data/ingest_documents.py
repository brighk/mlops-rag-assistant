"""
Ingest documents and create vector embeddings
"""
import os
import yaml
from pathlib import Path
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

class DocumentIngestor:
    """Ingest documents into vector database"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize embeddings
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config['embeddings']['model_name'],
            model_kwargs={'device': self.config['embeddings']['device']}
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config['rag']['chunk_size'],
            chunk_overlap=self.config['rag']['chunk_overlap'],
            length_function=len,
        )
        
    def load_documents(self, directory: str = "data/raw") -> List:
        """Load documents from directory"""
        print(f"\nLoading documents from {directory}...")
        
        documents = []
        
        # Load text files
        txt_loader = DirectoryLoader(
            directory,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )
        documents.extend(txt_loader.load())
        
        # Load PDFs
        pdf_files = list(Path(directory).rglob("*.pdf"))
        for pdf_path in tqdm(pdf_files, desc="Loading PDFs"):
            try:
                loader = PyPDFLoader(str(pdf_path))
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {pdf_path}: {e}")
        
        print(f"✓ Loaded {len(documents)} documents")
        return documents
    
    def create_vector_store(self, documents: List):
        """Create vector store from documents"""
        print("\nSplitting documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"✓ Created {len(chunks)} chunks")
        
        print("\nCreating vector store...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.config['vector_store']['persist_directory'],
            collection_name=self.config['vector_store']['collection_name']
        )
        
        print(f"✓ Vector store created with {len(chunks)} embeddings")
        return vector_store
    
    def ingest(self):
        """Main ingestion pipeline"""
        # Create data directories
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        Path("data/vector_store").mkdir(parents=True, exist_ok=True)
        
        # Load documents
        documents = self.load_documents()
        
        if not documents:
            print("\n⚠️  No documents found!")
            print("Add .txt or .pdf files to data/raw/ directory")
            return
        
        # Create vector store
        vector_store = self.create_vector_store(documents)
        
        print("\n" + "="*50)
        print("✓ Document ingestion complete!")
        print("="*50)

if __name__ == "__main__":
    ingestor = DocumentIngestor()
    ingestor.ingest()
