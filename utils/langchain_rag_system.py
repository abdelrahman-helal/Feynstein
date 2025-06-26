import os
import json
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

class LangChainRAGSystem:
    def __init__(self, index_name: str = "feynstein-db"):
        self.index_name = index_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        # Pinecone API key from environment
        self.pinecone_api_key = os.getenv("pinecone_KEY")
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key not found in environment variable 'pinecone_KEY'.")
        self.pinecone_client = Pinecone(api_key=self.pinecone_api_key)
    
    def load_textbooks(self, textbooks_dir: str = "textbooks") -> List[Document]:
        """Load all PDF textbooks from the specified directory"""
        documents = []
        
        if not os.path.exists(textbooks_dir):
            print(f"Textbooks directory {textbooks_dir} not found!")
            return documents
        
        for filename in os.listdir(textbooks_dir):
            if filename.endswith('.pdf'):
                file_path = os.path.join(textbooks_dir, filename)
                try:
                    print(f"Loading {filename}...")
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    # Add source information
                    for doc in docs:
                        doc.metadata['source'] = filename
                    documents.extend(docs)
                    print(f"Loaded {len(docs)} pages from {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
        
        return documents
    
    def process_and_save_vector_db(self, textbooks_dir: str = "textbooks") -> None:
        """Process textbooks and upload to Pinecone index"""
        print("Loading textbooks...")
        documents = self.load_textbooks(textbooks_dir)
        if not documents:
            print("No documents found to process!")
            return
        print(f"Processing {len(documents)} documents...")
        print("Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        # Use existing Pinecone index
        # pc = Pinecone(api_key=self.pinecone_api_key)
        if self.index_name not in [idx.name for idx in self.pinecone_client.list_indexes()]:
            print(f"Pinecone index '{self.index_name}' not found! Please create it first on the Pinecone dashboard.")
            return
        index = self.pinecone_client.Index(self.index_name)
        # Create Pinecone vector store
        self.vector_store = PineconeVectorStore(index=index, embedding=self.embeddings)
        from uuid import uuid4
        ids = [str(uuid4()) for _ in range(len(chunks))]
        print("Uploading chunks to Pinecone in batches...")
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch_docs = chunks[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            self.vector_store.add_documents(documents=batch_docs, ids=batch_ids)
        print(f"Uploaded {len(chunks)} chunks to Pinecone index '{self.index_name}'!")
    
    def load_vector_db(self) -> bool:
        """Load the Pinecone vector store"""
        try:
            if not self.pinecone_api_key:
                raise ValueError("Pinecone API key not found in environment variable 'pinecone_KEY'.")
            # pc = pinecone.Pinecone(api_key=self.pinecone_api_key)
            if self.index_name not in [idx.name for idx in self.pinecone_client.list_indexes()]:
                print(f"Pinecone index '{self.index_name}' not found!")
                return False
            index = self.pinecone_client.Index(self.index_name)
            self.vector_store = PineconeVectorStore(index=index, embedding=self.embeddings)
            print(f"Loaded Pinecone vector store for index '{self.index_name}'")
            return True
        except Exception as e:
            print(f"Error loading Pinecone vector store: {str(e)}")
            return False
    
    def query_relevant_content(self, query: str, k: int = 5) -> str:
        """Query the vector database for relevant content"""
        if self.vector_store is None:
            print("Vector store not initialized!")
            return ""
        
        try:
            # Search for relevant documents
            docs = self.vector_store.similarity_search(query, k=k)
            
            # Combine relevant content
            relevant_content = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                content = doc.page_content.strip()
                relevant_content.append(f"Source {i} ({source}):\n{content}\n")
            
            return "\n".join(relevant_content)
            
        except Exception as e:
            print(f"Error querying vector database: {str(e)}")
            return ""
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """Get relevant documents for a query (for use as a retriever)"""
        if self.vector_store is None:
            return []
        
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return []

class LangChainRetriever(BaseRetriever):
    """Custom retriever for LangChain integration"""
    
    def __init__(self, rag_system: LangChainRAGSystem, k: int = 5):
        self.rag_system = rag_system
        self.k = k
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.rag_system.get_relevant_documents(query, self.k)

# Create a singleton instance
langchain_rag = LangChainRAGSystem() 