"""
Retriever Module
Handles query processing and document retrieval from vector stores.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from embeddings import EmbeddingGenerator
from vector_store import VectorStore


class DocumentRetriever:
    """
    Retrieves relevant documents for user queries.
    Implements various retrieval strategies.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        strategy: str = "similarity"
    ):
        """
        Initialize Document Retriever.
        
        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score
            strategy: Retrieval strategy (similarity, mmr, threshold)
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.strategy = strategy
        
        logger.info(f"DocumentRetriever initialized with strategy: {strategy}, top_k: {top_k}")
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess user query.
        
        Args:
            query: Raw user query
            
        Returns:
            Preprocessed query
        """
        # Basic preprocessing
        query = query.strip()
        
        # Could add more sophisticated preprocessing:
        # - Expand acronyms
        # - Remove stopwords (optional, as embeddings handle this)
        # - Query expansion
        
        logger.debug(f"Preprocessed query: {query}")
        return query
    
    def retrieve(
        self,
        query: str,
        filter_metadata: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            filter_metadata: Optional metadata filters
            top_k: Override default top_k
            
        Returns:
            List of retrieved documents with metadata
        """
        logger.info(f"Retrieving documents for query: {query[:50]}...")
        
        # Preprocess query
        processed_query = self.preprocess_query(query)
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(processed_query)
        
        # Retrieve from vector store
        k = top_k or self.top_k
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=k,
            filter_metadata=filter_metadata
        )
        
        # Apply similarity threshold
        if self.strategy == "threshold":
            results = self._filter_by_threshold(results)
        
        # Apply MMR if specified
        elif self.strategy == "mmr":
            results = self._apply_mmr(results, query_embedding, k)
        
        logger.info(f"Retrieved {len(results)} documents")
        return results
    
    def _filter_by_threshold(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter results by similarity threshold.
        
        Args:
            results: Retrieved documents
            
        Returns:
            Filtered documents
        """
        filtered = []
        for result in results:
            # Check if score/distance meets threshold
            score = result.get('score')
            distance = result.get('distance')
            
            # Handle different similarity metrics
            if score is not None:
                # Higher score is better (cosine similarity)
                if score >= self.similarity_threshold:
                    filtered.append(result)
            elif distance is not None:
                # Lower distance is better (L2 distance)
                # Convert to similarity-like metric
                similarity = 1 / (1 + distance)
                if similarity >= self.similarity_threshold:
                    filtered.append(result)
            else:
                # No score/distance available, keep result
                filtered.append(result)
        
        logger.debug(f"Filtered from {len(results)} to {len(filtered)} documents")
        return filtered
    
    def _apply_mmr(
        self,
        results: List[Dict[str, Any]],
        query_embedding: List[float],
        k: int,
        lambda_mult: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Apply Maximal Marginal Relevance to diversify results.
        
        Args:
            results: Retrieved documents
            query_embedding: Query embedding vector
            k: Number of documents to return
            lambda_mult: Trade-off between relevance and diversity (0-1)
            
        Returns:
            Re-ranked documents
        """
        if len(results) <= k:
            return results
        
        # MMR implementation would require computing similarities
        # between all result pairs, which requires access to embeddings
        # For now, return top-k (can be enhanced later)
        logger.debug("MMR not fully implemented, returning top results")
        return results[:k]
    
    def retrieve_with_scores(
        self,
        query: str,
        filter_metadata: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> List[tuple]:
        """
        Retrieve documents with similarity scores.
        
        Args:
            query: User query
            filter_metadata: Optional metadata filters
            top_k: Override default top_k
            
        Returns:
            List of (document, score) tuples
        """
        results = self.retrieve(query, filter_metadata, top_k)
        
        # Extract scores
        scored_results = []
        for result in results:
            score = result.get('score', result.get('distance', 0))
            scored_results.append((result, score))
        
        return scored_results
    
    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            results: Retrieved documents
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            text = result.get('text', '')
            metadata = result.get('metadata', {})
            source = metadata.get('source', 'Unknown')
            
            context_parts.append(
                f"[Document {i}] (Source: {source})\n{text}\n"
            )
        
        context = "\n".join(context_parts)
        logger.debug(f"Formatted context with {len(results)} documents")
        return context
    
    def get_relevant_chunks(
        self,
        query: str,
        format_as_context: bool = True
    ) -> Dict[str, Any]:
        """
        High-level method to get relevant chunks for a query.
        
        Args:
            query: User query
            format_as_context: Whether to format as context string
            
        Returns:
            Dictionary with chunks and context
        """
        results = self.retrieve(query)
        
        response = {
            'query': query,
            'chunks': results,
            'count': len(results)
        }
        
        if format_as_context:
            response['context'] = self.format_context(results)
        
        return response


def main():
    """Demo/test function."""
    from embeddings import EmbeddingGenerator
    from vector_store import ChromaDBStore
    
    print(f"\n{'='*80}")
    print("Document Retriever Demo")
    print(f"{'='*80}\n")
    
    # Initialize components
    embedding_gen = EmbeddingGenerator(provider="openai")
    vector_store = ChromaDBStore(collection_name="test_retrieval")
    
    # Add some test documents
    test_docs = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand text.",
        "Computer vision allows machines to interpret images and videos.",
        "Reinforcement learning trains agents through rewards and penalties."
    ]
    
    print("Adding test documents...")
    embeddings = embedding_gen.generate_embeddings(test_docs)
    vector_store.add_documents(test_docs, embeddings)
    print(f"Added {len(test_docs)} documents\n")
    
    # Initialize retriever
    retriever = DocumentRetriever(
        vector_store=vector_store,
        embedding_generator=embedding_gen,
        top_k=3,
        strategy="similarity"
    )
    
    # Test queries
    queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "Tell me about NLP"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 80)
        
        results = retriever.retrieve(query)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['text']}")
            score = result.get('score', result.get('distance', 'N/A'))
            print(f"   Score: {score}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

