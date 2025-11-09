"""
Document Processing Script
Standalone script for processing documents and adding them to the vector store.
"""

import os
import sys
import argparse
from pathlib import Path
from loguru import logger

from document_processor import DocumentProcessor
from embeddings import EmbeddingGenerator
from vector_store import create_vector_store


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    logger.remove()
    
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")
    
    # Also log to file
    logger.add("logs/document_processing.log", rotation="10 MB")


def process_single_document(
    document_path: str,
    embedding_generator: EmbeddingGenerator,
    vector_store,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """Process a single document."""
    logger.info(f"Processing document: {document_path}")
    
    # Initialize document processor
    processor = DocumentProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Process document
    chunks = processor.process_document(document_path)
    
    logger.info(f"Generated {len(chunks)} chunks")
    
    # Generate embeddings
    texts = [chunk['text'] for chunk in chunks]
    logger.info("Generating embeddings...")
    embeddings = embedding_generator.generate_embeddings(texts)
    
    # Store in vector database
    logger.info("Storing in vector database...")
    metadata = [chunk['metadata'] for chunk in chunks]
    chunk_ids = [f"{Path(document_path).stem}_chunk_{i}" for i in range(len(chunks))]
    
    vector_store.add_documents(
        texts=texts,
        embeddings=embeddings,
        metadata=metadata,
        ids=chunk_ids
    )
    
    logger.info(f"Successfully processed document: {document_path}")
    return len(chunks)


def process_directory(
    directory_path: str,
    pattern: str,
    embedding_generator: EmbeddingGenerator,
    vector_store,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """Process all documents in a directory."""
    directory = Path(directory_path)
    
    if not directory.exists():
        logger.error(f"Directory not found: {directory}")
        return
    
    # Find all matching files
    files = list(directory.glob(pattern))
    logger.info(f"Found {len(files)} files matching pattern '{pattern}'")
    
    total_chunks = 0
    success_count = 0
    
    for file_path in files:
        try:
            chunks = process_single_document(
                str(file_path),
                embedding_generator,
                vector_store,
                chunk_size,
                chunk_overlap
            )
            total_chunks += chunks
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
    
    logger.info(f"Processed {success_count}/{len(files)} documents, total chunks: {total_chunks}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Process documents for RAG system")
    parser.add_argument("input", help="Input document or directory path")
    parser.add_argument("--pattern", default="*.pdf", help="File pattern for directory processing")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap")
    parser.add_argument("--vector-store", default="chromadb", choices=["chromadb", "pinecone", "faiss"])
    parser.add_argument("--collection", default="rag_documents", help="Collection/index name")
    parser.add_argument("--embedding-model", default="text-embedding-3-small", help="Embedding model")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    logger.info("Document Processing Script")
    logger.info("=" * 80)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Initialize components
    logger.info("Initializing components...")
    
    # Embedding generator
    embedding_gen = EmbeddingGenerator(
        provider="openai",
        model=args.embedding_model
    )
    
    # Vector store
    if args.vector_store == "chromadb":
        vector_store = create_vector_store(
            "chromadb",
            collection_name=args.collection
        )
    elif args.vector_store == "pinecone":
        vector_store = create_vector_store(
            "pinecone",
            index_name=args.collection
        )
    else:
        vector_store = create_vector_store("faiss")
    
    logger.info("Components initialized")
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        process_single_document(
            str(input_path),
            embedding_gen,
            vector_store,
            args.chunk_size,
            args.chunk_overlap
        )
    elif input_path.is_dir():
        process_directory(
            str(input_path),
            args.pattern,
            embedding_gen,
            vector_store,
            args.chunk_size,
            args.chunk_overlap
        )
    else:
        logger.error(f"Invalid input path: {input_path}")
        sys.exit(1)
    
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()

