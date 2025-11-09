# RAG System with AgentKit and n8n

A comprehensive Retrieval Augmented Generation (RAG) system integrating AgentKit and n8n workflows for intelligent document processing and query answering.

## 🎯 Project Overview

This project demonstrates an end-to-end RAG pipeline that:
- Extracts text from PDF documents
- Chunks text into manageable segments
- Generates embeddings using OpenAI
- Stores vectors in a vector database (Pinecone/ChromaDB/FAISS)
- Retrieves relevant context for user queries
- Generates accurate answers using LLM
- Integrates with AgentKit for autonomous agent capabilities
- Orchestrates workflows using n8n

## 📋 Project Specifications

### Document Source
- **Type**: PDF Documents
- **Support**: Multi-page PDFs, text-based and OCR-capable

### Text Extraction Method
- **Library**: PyPDF2 and pdfplumber for robust extraction
- **Fallback**: OCR using pytesseract for scanned documents
- **Preprocessing**: Text cleaning, normalization, and formatting

### Chunk Size
- **Size**: 500-1000 characters per chunk
- **Overlap**: 100-200 characters for context continuity
- **Method**: Recursive character splitting with semantic awareness

### Embedding Model
- **Primary**: OpenAI text-embedding-3-small (1536 dimensions)
- **Alternative**: text-embedding-ada-002
- **Local Option**: sentence-transformers for offline usage

### Vector Database
- **Primary**: Pinecone (cloud-based, production-ready)
- **Alternative 1**: ChromaDB (local, easy setup)
- **Alternative 2**: FAISS (high-performance, local)

### User Query Handling
- **Query Processing**: Natural language understanding with preprocessing
- **Retrieval Strategy**: Semantic similarity search (top-k results)
- **Re-ranking**: MMR (Maximal Marginal Relevance) for diversity

### Context Generation
- **LLM Model**: OpenAI GPT-4 or GPT-3.5-turbo
- **Prompt Engineering**: Context-aware prompts with retrieved chunks
- **Response Format**: Structured answers with source citations

### AgentKit Integration
- **Agent Creation**: Custom RAG agent with tools
- **Tools**: Document retrieval, summarization, Q&A
- **Orchestration**: Multi-step reasoning and task planning

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  PDF Input  │────▶│   Text       │────▶│   Chunking  │
│             │     │  Extraction  │     │             │
└─────────────┘     └──────────────┘     └─────────────┘
                                                 │
                                                 ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Vector    │◀────│   Embedding  │◀────│  Processed  │
│  Database   │     │  Generation  │     │   Chunks    │
└─────────────┘     └──────────────┘     └─────────────┘
      │
      │                                    ┌─────────────┐
      │                                    │User Query   │
      │                                    └──────┬──────┘
      │                                           │
      ▼                                           ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Retrieve   │────▶│   Context    │────▶│    LLM      │
│  Similar    │     │  Formation   │     │  Generation │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
                                                 ▼
                                          ┌─────────────┐
                                          │   Answer    │
                                          └─────────────┘
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
python --version

# Node.js 16+ (for n8n)
node --version
```

### Installation

1. **Clone and Setup Python Environment**

```bash
cd Rag_system
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Configure Environment Variables**

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your API keys
# - OPENAI_API_KEY
# - PINECONE_API_KEY (if using Pinecone)
```

4. **Install n8n (Optional)**

```bash
npm install -g n8n
```

### Running the System

#### 1. Process Documents

```bash
python src/process_documents.py --input data/sample.pdf
```

#### 2. Start RAG API Server

```bash
python src/api_server.py
```

#### 3. Start n8n Workflow

```bash
n8n start
# Import workflows from n8n_workflows/
```

#### 4. Run Demo

```bash
python demo.py
```

## 📁 Project Structure

```
Rag_system/
├── data/
│   ├── sample.pdf              # Sample PDF for demo
│   └── processed/              # Processed chunks
├── src/
│   ├── document_processor.py   # PDF extraction & chunking
│   ├── embeddings.py           # Embedding generation
│   ├── vector_store.py         # Vector database operations
│   ├── retriever.py            # Query & retrieval logic
│   ├── generator.py            # LLM answer generation
│   ├── rag_pipeline.py         # Complete RAG pipeline
│   ├── agentkit_integration.py # AgentKit agent setup
│   └── api_server.py           # FastAPI server
├── n8n_workflows/
│   ├── rag_workflow.json       # Main n8n workflow
│   └── document_processing.json # Document ingestion workflow
├── tests/
│   ├── test_document_processor.py
│   ├── test_embeddings.py
│   └── test_rag_pipeline.py
├── notebooks/
│   └── RAG_Demo.ipynb          # Interactive demo
├── requirements.txt
├── .env.example
├── config.yaml
├── demo.py
└── README.md
```

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
document_processing:
  chunk_size: 1000
  chunk_overlap: 200
  
embeddings:
  model: "text-embedding-3-small"
  dimensions: 1536
  
vector_store:
  type: "chromadb"  # pinecone, chromadb, faiss
  
retrieval:
  top_k: 5
  similarity_threshold: 0.7
  
generation:
  model: "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 500
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📊 Usage Examples

### Python API

```python
from src.rag_pipeline import RAGPipeline

# Initialize pipeline
rag = RAGPipeline(config_path="config.yaml")

# Process documents
rag.ingest_document("data/sample.pdf")

# Query
response = rag.query("What is the main topic of the document?")
print(response['answer'])
print(response['sources'])
```

### REST API

```bash
# Query endpoint
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?"}'
```

### n8n Workflow

1. Import `n8n_workflows/rag_workflow.json`
2. Configure webhook trigger
3. Connect to RAG API
4. Activate workflow

## 🤖 AgentKit Integration

The system includes an intelligent agent that can:
- Understand complex queries
- Retrieve relevant information
- Synthesize answers from multiple sources
- Perform multi-step reasoning
- Execute tool chains automatically

```python
from src.agentkit_integration import RAGAgent

agent = RAGAgent()
response = agent.run("Analyze the key points in the document and summarize them")
```

## 📈 Performance Optimization

- **Caching**: Query results cached for faster responses
- **Batch Processing**: Bulk document ingestion support
- **Async Operations**: Non-blocking I/O for API calls
- **Index Optimization**: Vector database indexing strategies

## 🔐 Security

- API key management via environment variables
- Rate limiting on API endpoints
- Input validation and sanitization
- Secure document storage

## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome! Please submit issues and pull requests.

## 📧 Support

For questions and support, please open an issue in the repository.

---

**Built with ❤️ for efficient knowledge retrieval and generation**

