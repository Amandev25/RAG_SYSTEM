"""
Vector Store Module
Manages vector database operations for storing and retrieving embeddings.
Supports Pinecone, ChromaDB, and FAISS.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
from loguru import logger

# Import vector database clients
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("Warning: chromadb not installed. Install with: pip install chromadb")

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    print("Warning: pinecone not installed. Install with: pip install pinecone-client")

try:
    import faiss
    import numpy as np
except ImportError:
    print("Warning: faiss not installed. Install with: pip install faiss-cpu")


class VectorStore:
    """Base class for vector store operations."""
    
    def __init__(self, store_type: str, **kwargs):
        """
        Initialize vector store.
        
        Args:
            store_type: Type of vector store (pinecone, chromadb, faiss)
            **kwargs: Additional configuration parameters
        """
        self.store_type = store_type
        self.config = kwargs
        logger.info(f"Initializing {store_type} vector store")
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ):
        """Add documents to vector store."""
        raise NotImplementedError
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        raise NotImplementedError
    
    def delete(self, ids: List[str]):
        """Delete documents by IDs."""
        raise NotImplementedError
    
    def clear(self):
        """Clear all documents from store."""
        raise NotImplementedError


class ChromaDBStore(VectorStore):
    """ChromaDB vector store implementation."""
    
    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./data/chroma_db",
        **kwargs
    ):
        super().__init__("chromadb", **kwargs)
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Create persist directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"ChromaDB initialized with collection: {collection_name}")
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ):
        """Add documents to ChromaDB."""
        if not texts or not embeddings:
            raise ValueError("texts and embeddings cannot be empty")
        
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have same length")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        # Prepare metadata
        if metadata is None:
            metadata = [{}] * len(texts)
        
        # Add to collection
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadata,
            ids=ids
        )
        
        logger.info(f"Added {len(texts)} documents to ChromaDB")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search ChromaDB for similar documents."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        logger.debug(f"Found {len(formatted_results)} results")
        return formatted_results
    
    def delete(self, ids: List[str]):
        """Delete documents from ChromaDB."""
        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents")
    
    def clear(self):
        """Clear all documents from collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Collection cleared")
    
    def get_count(self) -> int:
        """Get number of documents in collection."""
        return self.collection.count()


class PineconeStore(VectorStore):
    """Pinecone vector store implementation."""
    
    def __init__(
        self,
        index_name: str = "rag-system-index",
        dimension: int = 1536,
        metric: str = "cosine",
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        **kwargs
    ):
        super().__init__("pinecone", **kwargs)
        
        self.index_name = index_name
        self.dimension = dimension
        
        # Get API credentials
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.environment = environment or os.getenv("PINECONE_ENVIRONMENT")
        
        if not self.api_key:
            raise ValueError("Pinecone API key not provided")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        
        # Create index if it doesn't exist
        if index_name not in [index.name for index in self.pc.list_indexes()]:
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logger.info(f"Created Pinecone index: {index_name}")
        
        # Connect to index
        self.index = self.pc.Index(index_name)
        logger.info(f"Connected to Pinecone index: {index_name}")
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ):
        """Add documents to Pinecone."""
        if not texts or not embeddings:
            raise ValueError("texts and embeddings cannot be empty")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        # Prepare metadata
        if metadata is None:
            metadata = [{}] * len(texts)
        
        # Add text to metadata
        for i, text in enumerate(texts):
            metadata[i]['text'] = text
        
        # Prepare vectors for upsert
        vectors = list(zip(ids, embeddings, metadata))
        
        # Upsert to Pinecone
        self.index.upsert(vectors=vectors)
        
        logger.info(f"Added {len(texts)} documents to Pinecone")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search Pinecone for similar documents."""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filter_metadata,
            include_metadata=True
        )
        
        # Format results
        formatted_results = []
        for match in results['matches']:
            formatted_results.append({
                'id': match['id'],
                'text': match['metadata'].get('text', ''),
                'metadata': match['metadata'],
                'score': match['score']
            })
        
        logger.debug(f"Found {len(formatted_results)} results")
        return formatted_results
    
    def delete(self, ids: List[str]):
        """Delete documents from Pinecone."""
        self.index.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents")
    
    def clear(self):
        """Clear all documents from index."""
        self.index.delete(delete_all=True)
        logger.info("Index cleared")


class FAISSStore(VectorStore):
    """FAISS vector store implementation."""
    
    def __init__(
        self,
        dimension: int = 1536,
        index_path: str = "./data/faiss_index",
        **kwargs
    ):
        super().__init__("faiss", **kwargs)
        
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        
        # Store metadata separately
        self.metadata_store = {}
        self.text_store = {}
        self.id_counter = 0
        
        # Load existing index if available
        self._load_index()
        
        logger.info(f"FAISS initialized with dimension: {dimension}")
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ):
        """Add documents to FAISS."""
        if not texts or not embeddings:
            raise ValueError("texts and embeddings cannot be empty")
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Store metadata and texts
        if metadata is None:
            metadata = [{}] * len(texts)
        
        for i, text in enumerate(texts):
            doc_id = self.id_counter
            self.text_store[doc_id] = text
            self.metadata_store[doc_id] = metadata[i]
            self.id_counter += 1
        
        # Save index
        self._save_index()
        
        logger.info(f"Added {len(texts)} documents to FAISS")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search FAISS for similar documents."""
        # Convert query to numpy array
        query_array = np.array([query_embedding]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_array, top_k)
        
        # Format results
        formatted_results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.text_store):
                continue
            
            formatted_results.append({
                'id': str(idx),
                'text': self.text_store[idx],
                'metadata': self.metadata_store[idx],
                'distance': float(distances[0][i])
            })
        
        logger.debug(f"Found {len(formatted_results)} results")
        return formatted_results
    
    def delete(self, ids: List[str]):
        """Delete documents from FAISS (not directly supported, requires rebuild)."""
        logger.warning("FAISS does not support direct deletion. Consider rebuilding index.")
    
    def clear(self):
        """Clear all documents from FAISS."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata_store = {}
        self.text_store = {}
        self.id_counter = 0
        self._save_index()
        logger.info("FAISS index cleared")
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        faiss.write_index(self.index, str(self.index_path / "index.faiss"))
        
        with open(self.index_path / "metadata.pkl", 'wb') as f:
            pickle.dump({
                'metadata_store': self.metadata_store,
                'text_store': self.text_store,
                'id_counter': self.id_counter
            }, f)
    
    def _load_index(self):
        """Load FAISS index and metadata from disk."""
        index_file = self.index_path / "index.faiss"
        metadata_file = self.index_path / "metadata.pkl"
        
        if index_file.exists() and metadata_file.exists():
            self.index = faiss.read_index(str(index_file))
            
            with open(metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.metadata_store = data['metadata_store']
                self.text_store = data['text_store']
                self.id_counter = data['id_counter']
            
            logger.info("Loaded existing FAISS index")


def create_vector_store(store_type: str, **kwargs) -> VectorStore:
    """
    Factory function to create vector store.
    
    Args:
        store_type: Type of vector store (chromadb, pinecone, faiss)
        **kwargs: Additional configuration parameters
        
    Returns:
        VectorStore instance
    """
    if store_type == "chromadb":
        return ChromaDBStore(**kwargs)
    elif store_type == "pinecone":
        return PineconeStore(**kwargs)
    elif store_type == "faiss":
        return FAISSStore(**kwargs)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")


def main():
    """Demo/test function."""
    print(f"\n{'='*80}")
    print("Vector Store Demo")
    print(f"{'='*80}\n")
    
    # Test data
    texts = [
        "Artificial intelligence is transforming technology.",
        "Machine learning algorithms learn from data.",
        "Natural language processing enables text understanding.",
        "Computer vision allows machines to interpret images."
    ]
    
    # Generate dummy embeddings (normally would use EmbeddingGenerator)
    embeddings = [[0.1] * 1536 for _ in range(len(texts))]
    
    metadata = [
        {"topic": "AI", "category": "technology"},
        {"topic": "ML", "category": "algorithms"},
        {"topic": "NLP", "category": "language"},
        {"topic": "CV", "category": "vision"}
    ]
    
    # Test ChromaDB
    print("Testing ChromaDB...")
    chroma_store = ChromaDBStore(collection_name="test_collection")
    chroma_store.add_documents(texts, embeddings, metadata)
    print(f"Documents in store: {chroma_store.get_count()}")
    
    # Search
    results = chroma_store.search(embeddings[0], top_k=2)
    print(f"Search results: {len(results)}")
    for result in results:
        print(f"  - {result['text'][:50]}...")
    
    print("\nVector store demo complete!")


if __name__ == "__main__":
    main()

