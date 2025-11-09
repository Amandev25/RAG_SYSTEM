"""
Document Processor Module
Handles PDF text extraction, chunking, and preprocessing.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

try:
    import PyPDF2
    from pdfplumber import open as pdf_open
except ImportError:
    print("Warning: PDF libraries not installed. Install with: pip install pypdf2 pdfplumber")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


class DocumentProcessor:
    """
    Processes documents for RAG pipeline.
    Extracts text, chunks it, and prepares for embedding.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        extraction_method: str = "pdfplumber"
    ):
        """
        Initialize Document Processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks for context continuity
            extraction_method: Method to use for text extraction (pypdf2, pdfplumber)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extraction_method = extraction_method
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        logger.info(f"DocumentProcessor initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Extracting text from {pdf_path} using {self.extraction_method}")
        
        try:
            if self.extraction_method == "pdfplumber":
                return self._extract_with_pdfplumber(pdf_path)
            elif self.extraction_method == "pypdf2":
                return self._extract_with_pypdf2(pdf_path)
            else:
                raise ValueError(f"Unknown extraction method: {self.extraction_method}")
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """Extract text using pdfplumber."""
        text = ""
        try:
            with pdf_open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n\n--- Page {page_num} ---\n\n{page_text}"
            
            logger.info(f"Extracted {len(text)} characters from {len(pdf.pages)} pages")
            return text
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            # Fallback to PyPDF2
            return self._extract_with_pypdf2(pdf_path)
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> str:
        """Extract text using PyPDF2."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
            
            logger.info(f"Extracted {len(text)} characters from {num_pages} pages")
            return text
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)]', '', text)
        
        # Normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        logger.debug(f"Preprocessed text to {len(text)} characters")
        return text
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        logger.info(f"Chunking text of {len(text)} characters")
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create chunk objects with metadata
        chunk_objects = []
        for i, chunk in enumerate(chunks):
            chunk_obj = {
                "text": chunk,
                "chunk_index": i,
                "char_count": len(chunk),
                "metadata": metadata or {}
            }
            chunk_objects.append(chunk_obj)
        
        logger.info(f"Created {len(chunk_objects)} chunks")
        return chunk_objects
    
    def process_document(
        self,
        document_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Complete document processing pipeline.
        
        Args:
            document_path: Path to document file
            metadata: Optional metadata for the document
            
        Returns:
            List of processed chunks with metadata
        """
        logger.info(f"Processing document: {document_path}")
        
        # Extract text
        raw_text = self.extract_text_from_pdf(document_path)
        
        # Preprocess
        clean_text = self.preprocess_text(raw_text)
        
        # Add document metadata
        doc_metadata = {
            "source": str(document_path),
            "filename": Path(document_path).name,
            **(metadata or {})
        }
        
        # Chunk text
        chunks = self.chunk_text(clean_text, metadata=doc_metadata)
        
        logger.info(f"Document processing complete: {len(chunks)} chunks created")
        return chunks
    
    def process_directory(
        self,
        directory_path: str,
        pattern: str = "*.pdf"
    ) -> List[Dict[str, Any]]:
        """
        Process all documents in a directory.
        
        Args:
            directory_path: Path to directory containing documents
            pattern: File pattern to match (default: *.pdf)
            
        Returns:
            List of all chunks from all documents
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        all_chunks = []
        pdf_files = list(directory.glob(pattern))
        
        logger.info(f"Found {len(pdf_files)} files matching pattern '{pattern}'")
        
        for pdf_file in pdf_files:
            try:
                chunks = self.process_document(str(pdf_file))
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")
                continue
        
        logger.info(f"Processed {len(pdf_files)} documents, total chunks: {len(all_chunks)}")
        return all_chunks


def main():
    """Demo/test function."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python document_processor.py <pdf_path>")
        return
    
    pdf_path = sys.argv[1]
    
    # Initialize processor
    processor = DocumentProcessor(
        chunk_size=1000,
        chunk_overlap=200,
        extraction_method="pdfplumber"
    )
    
    # Process document
    chunks = processor.process_document(pdf_path)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"Document: {pdf_path}")
    print(f"Total chunks: {len(chunks)}")
    print(f"{'='*80}\n")
    
    # Show first 3 chunks
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\nChunk {i}:")
        print(f"Characters: {chunk['char_count']}")
        print(f"Text preview: {chunk['text'][:200]}...")
        print("-" * 80)


if __name__ == "__main__":
    main()

