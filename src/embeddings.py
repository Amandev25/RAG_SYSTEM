"""
Embeddings Module
Handles embedding generation using OpenAI and other providers.
"""

import os
from typing import List, Optional, Union
import numpy as np
from loguru import logger

try:
    from openai import OpenAI
except ImportError:
    print("Warning: OpenAI not installed. Install with: pip install openai")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")


class EmbeddingGenerator:
    """
    Generates embeddings for text using various providers.
    Supports OpenAI, HuggingFace, and local models.
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        batch_size: int = 100
    ):
        """
        Initialize Embedding Generator.
        
        Args:
            provider: Embedding provider (openai, huggingface)
            model: Model name
            api_key: API key (for OpenAI)
            batch_size: Batch size for processing
        """
        self.provider = provider
        self.model = model
        self.batch_size = batch_size
        
        # Initialize provider
        if provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"Initialized OpenAI embeddings with model: {model}")
        
        elif provider == "huggingface":
            self.client = SentenceTransformer(model)
            logger.info(f"Initialized HuggingFace embeddings with model: {model}")
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            if self.provider == "openai":
                return self._generate_openai_embedding(text)
            elif self.provider == "huggingface":
                return self._generate_huggingface_embedding(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            if self.provider == "openai":
                batch_embeddings = self._generate_openai_embeddings_batch(batch)
            elif self.provider == "huggingface":
                batch_embeddings = self._generate_huggingface_embeddings_batch(batch)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
            
            all_embeddings.extend(batch_embeddings)
            
            logger.debug(f"Processed batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
        
        logger.info(f"Generated {len(all_embeddings)} embeddings")
        return all_embeddings
    
    def _generate_openai_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding
    
    def _generate_openai_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings batch using OpenAI API."""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [item.embedding for item in response.data]
    
    def _generate_huggingface_embedding(self, text: str) -> List[float]:
        """Generate embedding using HuggingFace model."""
        embedding = self.client.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def _generate_huggingface_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings batch using HuggingFace model."""
        embeddings = self.client.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        if self.provider == "openai":
            # Known dimensions for OpenAI models
            dimensions_map = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536
            }
            return dimensions_map.get(self.model, 1536)
        
        elif self.provider == "huggingface":
            # Get dimension from model
            test_embedding = self.generate_embedding("test")
            return len(test_embedding)
        
        return 1536  # default
    
    def calculate_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0-1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(similarity)


class EmbeddingCache:
    """Simple in-memory cache for embeddings."""
    
    def __init__(self):
        self.cache = {}
        logger.info("Initialized embedding cache")
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        return self.cache.get(text)
    
    def set(self, text: str, embedding: List[float]):
        """Store embedding in cache."""
        self.cache[text] = embedding
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)


def main():
    """Demo/test function."""
    import sys
    
    # Test texts
    texts = [
        "What is artificial intelligence?",
        "Machine learning is a subset of AI.",
        "Natural language processing enables computers to understand human language.",
        "Python is a popular programming language for data science."
    ]
    
    # Initialize generator
    generator = EmbeddingGenerator(
        provider="openai",
        model="text-embedding-3-small"
    )
    
    print(f"\n{'='*80}")
    print(f"Embedding Generator Demo")
    print(f"Provider: {generator.provider}")
    print(f"Model: {generator.model}")
    print(f"Embedding Dimension: {generator.get_embedding_dimension()}")
    print(f"{'='*80}\n")
    
    # Generate embeddings
    embeddings = generator.generate_embeddings(texts)
    
    print(f"Generated {len(embeddings)} embeddings\n")
    
    # Show first embedding (truncated)
    print(f"Example embedding (first 10 values):")
    print(embeddings[0][:10])
    print(f"...\n")
    
    # Calculate similarities
    print(f"Similarity Matrix:")
    print(f"{'-'*60}")
    for i, text1 in enumerate(texts):
        for j, text2 in enumerate(texts):
            if i < j:
                similarity = generator.calculate_similarity(embeddings[i], embeddings[j])
                print(f"Text {i+1} vs Text {j+1}: {similarity:.4f}")
    print(f"{'-'*60}\n")


if __name__ == "__main__":
    main()

