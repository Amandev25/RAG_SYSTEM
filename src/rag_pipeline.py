"""
RAG Pipeline Module
Complete end-to-end RAG (Retrieval Augmented Generation) pipeline.
"""

import os
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from document_processor import DocumentProcessor
from embeddings import EmbeddingGenerator
from vector_store import create_vector_store, VectorStore
from retriever import DocumentRetriever
from generator import AnswerGenerator


class RAGPipeline:
    """
    Complete RAG pipeline integrating document processing,
    embedding, retrieval, and generation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize RAG Pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self._initialize_components()
        
        logger.info("RAG Pipeline initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        else:
            logger.warning("No config file provided, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'document_processing': {
                'chunking': {
                    'chunk_size': 1000,
                    'chunk_overlap': 200
                }
            },
            'embeddings': {
                'provider': 'openai',
                'model': 'text-embedding-3-small'
            },
            'vector_store': {
                'type': 'chromadb',
                'chromadb': {
                    'persist_directory': './data/chroma_db',
                    'collection_name': 'rag_documents'
                }
            },
            'retrieval': {
                'top_k': 5,
                'similarity_threshold': 0.7,
                'strategy': 'similarity'
            },
            'generation': {
                'model': 'gpt-3.5-turbo',
                'temperature': 0.7,
                'max_tokens': 500
            }
        }
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        # Document Processor
        doc_config = self.config['document_processing']['chunking']
        self.document_processor = DocumentProcessor(
            chunk_size=doc_config['chunk_size'],
            chunk_overlap=doc_config['chunk_overlap']
        )
        logger.info("✓ Document processor initialized")
        
        # Embedding Generator
        emb_config = self.config['embeddings']
        self.embedding_generator = EmbeddingGenerator(
            provider=emb_config['provider'],
            model=emb_config['model']
        )
        logger.info("✓ Embedding generator initialized")
        
        # Vector Store
        vs_config = self.config['vector_store']
        store_type = vs_config['type']
        store_params = vs_config.get(store_type, {})
        self.vector_store = create_vector_store(store_type, **store_params)
        logger.info("✓ Vector store initialized")
        
        # Document Retriever
        ret_config = self.config['retrieval']
        self.retriever = DocumentRetriever(
            vector_store=self.vector_store,
            embedding_generator=self.embedding_generator,
            top_k=ret_config['top_k'],
            similarity_threshold=ret_config['similarity_threshold'],
            strategy=ret_config['strategy']
        )
        logger.info("✓ Document retriever initialized")
        
        # Answer Generator
        gen_config = self.config['generation']
        self.generator = AnswerGenerator(
            model=gen_config['model'],
            temperature=gen_config['temperature'],
            max_tokens=gen_config['max_tokens']
        )
        logger.info("✓ Answer generator initialized")
    
    def ingest_document(
        self,
        document_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest a document into the RAG system.
        
        Args:
            document_path: Path to document file
            metadata: Optional metadata for the document
            
        Returns:
            Ingestion status and statistics
        """
        logger.info(f"Ingesting document: {document_path}")
        
        try:
            # Process document
            chunks = self.document_processor.process_document(
                document_path,
                metadata=metadata
            )
            
            # Generate embeddings
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedding_generator.generate_embeddings(texts)
            
            # Store in vector database
            chunk_metadata = [chunk['metadata'] for chunk in chunks]
            chunk_ids = [f"{Path(document_path).stem}_chunk_{i}" for i in range(len(chunks))]
            
            self.vector_store.add_documents(
                texts=texts,
                embeddings=embeddings,
                metadata=chunk_metadata,
                ids=chunk_ids
            )
            
            result = {
                'status': 'success',
                'document': document_path,
                'chunks_processed': len(chunks),
                'embeddings_generated': len(embeddings)
            }
            
            logger.info(f"Successfully ingested document with {len(chunks)} chunks")
            return result
            
        except Exception as e:
            logger.error(f"Error ingesting document: {e}")
            return {
                'status': 'error',
                'document': document_path,
                'error': str(e)
            }
    
    def ingest_directory(
        self,
        directory_path: str,
        pattern: str = "*.pdf"
    ) -> Dict[str, Any]:
        """
        Ingest all documents from a directory.
        
        Args:
            directory_path: Path to directory
            pattern: File pattern to match
            
        Returns:
            Ingestion statistics
        """
        logger.info(f"Ingesting directory: {directory_path}")
        
        directory = Path(directory_path)
        files = list(directory.glob(pattern))
        
        results = []
        success_count = 0
        error_count = 0
        total_chunks = 0
        
        for file_path in files:
            result = self.ingest_document(str(file_path))
            results.append(result)
            
            if result['status'] == 'success':
                success_count += 1
                total_chunks += result['chunks_processed']
            else:
                error_count += 1
        
        return {
            'status': 'completed',
            'total_files': len(files),
            'successful': success_count,
            'errors': error_count,
            'total_chunks': total_chunks,
            'results': results
        }
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_sources: bool = True,
        return_context: bool = False
    ) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve
            return_sources: Whether to include source information
            return_context: Whether to include retrieved context
            
        Returns:
            Answer and metadata
        """
        logger.info(f"Processing query: {question}")
        
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(question, top_k=top_k)
            
            if not retrieved_docs:
                return {
                    'question': question,
                    'answer': "I couldn't find any relevant information to answer your question.",
                    'sources': [],
                    'num_sources': 0
                }
            
            # Format context
            context = self.retriever.format_context(retrieved_docs)
            
            # Generate answer
            generation_result = self.generator.generate_answer(
                question=question,
                context=context,
                include_sources=return_sources
            )
            
            # Build response
            response = {
                'question': question,
                'answer': generation_result['answer'],
                'num_sources': len(retrieved_docs),
                'model': generation_result['model'],
                'usage': generation_result.get('usage', {})
            }
            
            if return_sources:
                response['sources'] = [
                    {
                        'text': doc['text'][:200] + '...',
                        'metadata': doc.get('metadata', {})
                    }
                    for doc in retrieved_docs
                ]
            
            if return_context:
                response['context'] = context
            
            logger.info("Query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'question': question,
                'answer': f"Error processing query: {str(e)}",
                'error': str(e)
            }
    
    def batch_query(
        self,
        questions: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        Args:
            questions: List of questions
            **kwargs: Additional parameters for query method
            
        Returns:
            List of responses
        """
        logger.info(f"Processing batch of {len(questions)} queries")
        
        results = []
        for question in questions:
            result = self.query(question, **kwargs)
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            'vector_store_type': self.vector_store.store_type,
            'embedding_model': self.embedding_generator.model,
            'generation_model': self.generator.model
        }
        
        # Add vector store specific stats
        if hasattr(self.vector_store, 'get_count'):
            stats['total_documents'] = self.vector_store.get_count()
        
        return stats
    
    def clear_database(self):
        """Clear all documents from vector store."""
        logger.warning("Clearing vector store...")
        self.vector_store.clear()
        logger.info("Vector store cleared")


def main():
    """Demo/test function."""
    import sys
    
    print(f"\n{'='*80}")
    print("RAG Pipeline Demo")
    print(f"{'='*80}\n")
    
    # Initialize pipeline
    pipeline = RAGPipeline(config_path="config.yaml")
    
    print("Pipeline initialized successfully!\n")
    print("Statistics:")
    stats = pipeline.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Check if document path provided
    if len(sys.argv) > 1:
        doc_path = sys.argv[1]
        print(f"\nIngesting document: {doc_path}")
        result = pipeline.ingest_document(doc_path)
        print(f"Result: {result}")
        
        # Test query
        test_query = "What is the main topic of this document?"
        print(f"\nTest query: {test_query}")
        response = pipeline.query(test_query)
        print(f"\nAnswer: {response['answer']}")
        print(f"Sources used: {response['num_sources']}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

