"""
API Server Module
FastAPI server for RAG system with REST endpoints.
"""

import os
from typing import Optional, List
from pathlib import Path
from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger
import uvicorn

from rag_pipeline import RAGPipeline


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(..., description="User's question")
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve")
    return_sources: bool = Field(True, description="Include source information")
    return_context: bool = Field(False, description="Include retrieved context")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    question: str
    answer: str
    num_sources: int
    sources: Optional[List[dict]] = None
    context: Optional[str] = None
    model: Optional[str] = None
    usage: Optional[dict] = None


class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    document_path: str = Field(..., description="Path to document file")
    metadata: Optional[dict] = Field(None, description="Document metadata")


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    status: str
    document: str
    chunks_processed: Optional[int] = None
    error: Optional[str] = None


class StatusResponse(BaseModel):
    """Response model for status endpoint."""
    status: str
    pipeline_initialized: bool
    statistics: dict


# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="REST API for Retrieval Augmented Generation system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG pipeline instance
pipeline: Optional[RAGPipeline] = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup."""
    global pipeline
    try:
        config_path = os.getenv("CONFIG_PATH", "config.yaml")
        pipeline = RAGPipeline(config_path=config_path)
        logger.info("RAG Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "RAG System API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipeline_initialized": pipeline is not None
    }


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status and statistics."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    stats = pipeline.get_statistics()
    
    return {
        "status": "operational",
        "pipeline_initialized": True,
        "statistics": stats
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Query the RAG system.
    
    Args:
        request: Query request with question and parameters
        
    Returns:
        Query response with answer and sources
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        logger.info(f"Received query: {request.question}")
        
        response = pipeline.query(
            question=request.question,
            top_k=request.top_k,
            return_sources=request.return_sources,
            return_context=request.return_context
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest a document into the RAG system.
    
    Args:
        request: Ingest request with document path and metadata
        background_tasks: FastAPI background tasks
        
    Returns:
        Ingestion status
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Verify document exists
        if not Path(request.document_path).exists():
            raise HTTPException(status_code=404, detail="Document not found")
        
        logger.info(f"Ingesting document: {request.document_path}")
        
        # Process in foreground for now (can be moved to background)
        result = pipeline.ingest_document(
            document_path=request.document_path,
            metadata=request.metadata
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/upload")
async def upload_and_ingest(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and ingest a document.
    
    Args:
        file: Uploaded file
        background_tasks: FastAPI background tasks
        
    Returns:
        Ingestion status
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Save uploaded file
        upload_dir = Path("data/documents")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Uploaded file: {file_path}")
        
        # Ingest document
        result = pipeline.ingest_document(str(file_path))
        
        return result
        
    except Exception as e:
        logger.error(f"Error uploading/ingesting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/directory")
async def ingest_directory_endpoint(
    directory_path: str,
    pattern: str = "*.pdf"
):
    """
    Ingest all documents from a directory.
    
    Args:
        directory_path: Path to directory
        pattern: File pattern to match
        
    Returns:
        Ingestion statistics
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        if not Path(directory_path).exists():
            raise HTTPException(status_code=404, detail="Directory not found")
        
        logger.info(f"Ingesting directory: {directory_path}")
        
        result = pipeline.ingest_directory(directory_path, pattern)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting directory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-query")
async def batch_query_endpoint(questions: List[str], top_k: int = 5):
    """
    Process multiple queries in batch.
    
    Args:
        questions: List of questions
        top_k: Number of documents to retrieve per question
        
    Returns:
        List of query responses
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        logger.info(f"Processing batch of {len(questions)} queries")
        
        results = pipeline.batch_query(questions, top_k=top_k)
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing batch queries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear")
async def clear_database():
    """Clear all documents from the vector store."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        logger.warning("Clearing database...")
        pipeline.clear_database()
        
        return {
            "status": "success",
            "message": "Database cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Error clearing database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False
):
    """
    Run the API server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload
    """
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    # Load configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    
    logger.info(f"Starting API server on {host}:{port}")
    run_server(host=host, port=port, reload=reload)

