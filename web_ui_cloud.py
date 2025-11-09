"""
RAG System - Cloud Web UI (Streamlit Cloud Compatible)
Uses HuggingFace Inference API instead of Ollama
"""

import sys
sys.path.insert(0, 'src')

import streamlit as st
from pathlib import Path
import time
import requests
import os

# Page configuration
st.set_page_config(
    page_title="RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: rgba(0, 0, 0, 0.6);
        margin-bottom: 2rem;
    }
    @media (prefers-color-scheme: dark) {
        .sub-header {
            color: rgba(255, 255, 255, 0.6);
        }
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
    }
    .source-box {
        background: rgba(30, 136, 229, 0.05);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(30, 136, 229, 0.2);
        border-left: 4px solid #1E88E5;
    }
    .answer-box {
        background: rgba(30, 136, 229, 0.08);
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-size: 1.1rem;
        line-height: 1.6;
        border: 1px solid rgba(30, 136, 229, 0.15);
    }
    .stat-card {
        background: rgba(30, 136, 229, 0.05);
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        border: 2px solid rgba(30, 136, 229, 0.3);
    }
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .stat-label {
        color: rgba(0, 0, 0, 0.6);
        font-size: 0.9rem;
    }
    @media (prefers-color-scheme: dark) {
        .stat-label {
            color: rgba(255, 255, 255, 0.6);
        }
    }
    .footer-text {
        text-align: center;
        color: rgba(0, 0, 0, 0.5);
        font-size: 0.9rem;
    }
    @media (prefers-color-scheme: dark) {
        .footer-text {
            color: rgba(255, 255, 255, 0.5);
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize ChromaDB and Embeddings
@st.cache_resource
def initialize_system():
    """Initialize the RAG system components"""
    from sentence_transformers import SentenceTransformer
    import chromadb
    
    # Initialize embedding model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="rag_documents",
        metadata={"hnsw:space": "cosine"}
    )
    
    return model, collection

def query_huggingface(prompt, hf_token=None):
    """Query HuggingFace API with multiple model fallbacks"""
    
    # Try multiple models in order of preference
    models = [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "microsoft/Phi-3-mini-4k-instruct",
        "google/flan-t5-large"
    ]
    
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.95,
            "return_full_text": False
        }
    }
    
    for model in models:
        try:
            API_URL = f"https://api-inference.huggingface.co/models/{model}"
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', 'No response generated')
                return str(result)
            elif response.status_code == 503:
                # Model is loading, try next one
                continue
            elif response.status_code != 410:
                # Not a "gone" error, might be rate limit, try next
                continue
        except Exception:
            continue
    
    return "⚠️ All models are currently unavailable. Please add a HuggingFace token for priority access or try again later."

# Initialize session state
if 'initialized' not in st.session_state:
    with st.spinner('🚀 Initializing RAG System...'):
        st.session_state.model, st.session_state.collection = initialize_system()
        st.session_state.initialized = True

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Header
st.markdown('<div class="main-header">🤖 RAG System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent Question Answering - Cloud Deployment</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 System Status")
    
    # Get HuggingFace token from secrets or input
    if 'hf_token' not in st.session_state:
        st.session_state.hf_token = st.secrets.get("HF_TOKEN", "") if hasattr(st, 'secrets') else ""
    
    if not st.session_state.hf_token:
        with st.expander("🔑 HuggingFace Token (Optional)", expanded=True):
            st.info("Enter your HF token for faster responses. [Get one free here](https://huggingface.co/settings/tokens)")
            hf_input = st.text_input("Token:", type="password", key="hf_input")
            if st.button("Save Token"):
                st.session_state.hf_token = hf_input
                st.success("✅ Token saved!")
                st.rerun()
    
    # Get statistics
    try:
        count = st.session_state.collection.count()
    except:
        count = 0
    
    # Display stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{count}</div>
            <div class="stat-label">Documents</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">FREE</div>
            <div class="stat-label">Cost</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🔧 System Info")
    st.info("""
    **Vector Store:** ChromaDB  
    **Embedding Model:** MiniLM-L6-v2  
    **LLM:** HuggingFace Mistral-7B  
    **Deployment:** Streamlit Cloud ☁️
    """)
    
    st.markdown("---")
    
    # PDF Upload section
    st.markdown("### 📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=['pdf', 'txt'],
        help="Upload a document to add to your knowledge base"
    )
    
    if uploaded_file is not None:
        if st.button("🚀 Process Document", type="primary", use_container_width=True):
            with st.spinner("📖 Processing document..."):
                try:
                    import tempfile
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Process document
                    from document_processor import DocumentProcessor
                    
                    processor = DocumentProcessor()
                    chunks = processor.process_document(tmp_path)
                    
                    if chunks:
                        # Add to vector store
                        embeddings = st.session_state.model.encode([chunk['text'] for chunk in chunks])
                        
                        st.session_state.collection.add(
                            documents=[chunk['text'] for chunk in chunks],
                            embeddings=embeddings.tolist(),
                            metadatas=[{
                                'source': uploaded_file.name,
                                'chunk_id': i
                            } for i in range(len(chunks))],
                            ids=[f"upload_{int(time.time())}_{i}" for i in range(len(chunks))]
                        )
                        
                        st.success(f"✅ Processed {uploaded_file.name}!")
                        st.info(f"📊 Added {len(chunks)} chunks")
                        
                        os.unlink(tmp_path)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Could not extract text")
                        os.unlink(tmp_path)
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    if st.button("🔄 Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main content
st.markdown("### 💬 Ask me anything!")

question = st.text_input(
    "Your question:",
    placeholder="e.g., What is machine learning?",
    key="question_input"
)

col1, col2 = st.columns([1, 4])

with col1:
    ask_button = st.button("🚀 Ask", type="primary", use_container_width=True)

# Process question
if ask_button and question:
    with st.spinner('🤔 Thinking...'):
        # Search for relevant documents
        query_emb = st.session_state.model.encode([question])
        results = st.session_state.collection.query(
            query_embeddings=query_emb.tolist(),
            n_results=3
        )
        
        # Build context
        context = "\n\n".join(results['documents'][0]) if results['documents'][0] else "No context available"
        
        # Generate answer
        prompt = f"""Based on the following context, answer the question. If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        answer = query_huggingface(prompt, st.session_state.hf_token)
        
        # Add to chat history
        st.session_state.chat_history.append({
            'question': question,
            'answer': answer,
            'sources': results['documents'][0][:3] if results['documents'] else []
        })

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    
    for chat in reversed(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**🙋 You asked:** {chat['question']}")
            
            st.markdown(f"""
            <div class="answer-box">
                <strong>🤖 Answer:</strong><br>
                {chat['answer']}
            </div>
            """, unsafe_allow_html=True)
            
            if chat['sources']:
                with st.expander(f"📚 View {len(chat['sources'])} source(s)"):
                    for j, source in enumerate(chat['sources'], 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Source {j}:</strong><br>
                            {source}
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("---")
else:
    st.info("""
    👋 **Welcome to the RAG System!**
    
    Upload a PDF document using the sidebar, then ask questions about it!
    
    Just type your question above and click "Ask" 🚀
    """)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer-text">
    🤖 RAG System | Deployed on Streamlit Cloud | 100% FREE ☁️
</div>
""", unsafe_allow_html=True)

