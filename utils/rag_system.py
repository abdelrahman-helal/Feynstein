from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import os
from pathlib import Path
import torch
from .equation_processor import equation_processor
import json
from datetime import datetime

class PhysicsRAGSystem:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(model_name, device=self.device)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            persist_directory="physics_textbooks_db",
            anonymized_telemetry=False
        ))
        
        # Create or get collections
        self.text_collection = self.chroma_client.get_or_create_collection(
            name="physics_textbooks",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.equation_collection = self.chroma_client.get_or_create_collection(
            name="physics_equations",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Add conversation history collection
        self.conversation_collection = self.chroma_client.get_or_create_collection(
            name="conversation_history",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Textbook metadata
        self.textbooks = {
            "openstax_vol1": "OpenStax University Physics Volume 1",
            "openstax_vol2": "OpenStax University Physics Volume 2",
            "openstax_vol3": "OpenStax University Physics Volume 3",
            "griffiths_qm": "Griffiths Introduction to Quantum Mechanics",
            "griffiths_em": "Griffiths Introduction to Electrodynamics",
            "taylor_cm": "Taylor Classical Mechanics"
        }
    
    def process_textbook(self, pdf_path: str, textbook_id: str):
        """
        Process a textbook PDF and add it to the vector database.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Textbook PDF not found: {pdf_path}")
        
        # Load and split the PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        chunks = self.text_splitter.split_documents(pages)
        
        # Process text and equations separately
        text_documents = []
        text_metadatas = []
        text_ids = []
        
        equation_documents = []
        equation_metadatas = []
        equation_ids = []
        
        for i, chunk in enumerate(chunks):
            content = chunk.page_content
            
            # Check if chunk contains equations
            if self._contains_equation(content):
                # Process equations in the chunk
                equations = self._extract_equations(content)
                for j, eq in enumerate(equations):
                    latex_eq, metadata = equation_processor.process_equation_text(eq)
                    equation_documents.append(latex_eq)
                    equation_metadatas.append({
                        "textbook": self.textbooks[textbook_id],
                        "page": chunk.metadata.get("page", 0),
                        "source": textbook_id,
                        "type": metadata["type"],
                        "variables": str(metadata["variables"]),
                        "operators": str(metadata["operators"]),
                        "complexity": metadata["complexity"]
                    })
                    equation_ids.append(f"{textbook_id}_eq_{i}_{j}")
            else:
                # Add to text collection
                text_documents.append(content)
                text_metadatas.append({
                    "textbook": self.textbooks[textbook_id],
                    "page": chunk.metadata.get("page", 0),
                    "source": textbook_id
                })
                text_ids.append(f"{textbook_id}_{i}")
        
        # Add to respective collections
        if text_documents:
            self.text_collection.add(
                documents=text_documents,
                metadatas=text_metadatas,
                ids=text_ids
            )
        
        if equation_documents:
            self.equation_collection.add(
                documents=equation_documents,
                metadatas=equation_metadatas,
                ids=equation_ids
            )
    
    def query_relevant_content(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector database for relevant content.
        """
        # Get query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Check if query is an equation
        if self._contains_equation(query):
            # Process equation
            latex_eq, metadata = equation_processor.process_equation_text(query)
            
            # Search in equation collection
            results = self.equation_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where={
                    "type": metadata["type"],
                    "complexity": {"$lte": metadata["complexity"] + 2}
                },
                include=["documents", "metadatas", "distances"]
            )
        else:
            # Search in text collection
            results = self.text_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
        
        # Format results
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "relevance_score": 1 - results["distances"][0][i]
            })
        
        return formatted_results
    
    def _contains_equation(self, text: str) -> bool:
        """
        Check if text contains mathematical equations.
        """
        # Look for common equation indicators
        equation_indicators = ['=', '∫', '∑', '∏', '∇', '∂']
        return any(indicator in text for indicator in equation_indicators)
    
    def _extract_equations(self, text: str) -> List[str]:
        """
        Extract equations from text.
        """
        # Split text by common equation separators
        equations = []
        current_eq = []
        
        for line in text.split('\n'):
            if self._contains_equation(line):
                current_eq.append(line)
            elif current_eq:
                equations.append('\n'.join(current_eq))
                current_eq = []
        
        if current_eq:
            equations.append('\n'.join(current_eq))
        
        return equations

    def add_to_conversation_history(self, chat_id: str, user_message: str, assistant_response: str):
        """
        Add a conversation turn to the history.
        
        Args:
            chat_id: Unique identifier for the chat session
            user_message: The user's message
            assistant_response: The assistant's response
        """
        timestamp = datetime.now().isoformat()
        conversation_turn = {
            "user_message": user_message,
            "assistant_response": assistant_response,
            "timestamp": timestamp
        }
        
        # Get existing conversation or create new one
        try:
            existing = self.conversation_collection.get(
                where={"chat_id": chat_id},
                include=["documents", "metadatas"]
            )
            if existing["documents"]:
                # Update existing conversation
                history = json.loads(existing["documents"][0])
                history.append(conversation_turn)
                self.conversation_collection.update(
                    ids=[existing["ids"][0]],
                    documents=[json.dumps(history)],
                    metadatas=[{"chat_id": chat_id, "last_updated": timestamp}]
                )
            else:
                # Create new conversation
                self.conversation_collection.add(
                    documents=[json.dumps([conversation_turn])],
                    metadatas=[{"chat_id": chat_id, "last_updated": timestamp}],
                    ids=[chat_id]
                )
        except Exception as e:
            # If no existing conversation, create new one
            self.conversation_collection.add(
                documents=[json.dumps([conversation_turn])],
                metadatas=[{"chat_id": chat_id, "last_updated": timestamp}],
                ids=[chat_id]
            )

    def get_conversation_history(self, chat_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        Retrieve conversation history for a specific chat.
        
        Args:
            chat_id: Unique identifier for the chat session
            limit: Maximum number of conversation turns to return
            
        Returns:
            List of conversation turns, each containing user_message and assistant_response
        """
        try:
            results = self.conversation_collection.get(
                where={"chat_id": chat_id},
                include=["documents"]
            )
            if results["documents"]:
                history = json.loads(results["documents"][0])
                return history[-limit:]  # Return the most recent turns
            return []
        except Exception:
            return []

    def clear_conversation_history(self, chat_id: str):
        """
        Clear conversation history for a specific chat.
        
        Args:
            chat_id: Unique identifier for the chat session
        """
        try:
            self.conversation_collection.delete(
                where={"chat_id": chat_id}
            )
        except Exception:
            pass  # Ignore if chat doesn't exist

# Create a singleton instance
physics_rag = PhysicsRAGSystem() 