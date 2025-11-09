"""
RAG System - Streamlit Cloud Compatible
Simplified version with minimal dependencies
"""

import streamlit as st
import requests
from pathlib import Path
import time
import sys

# Page configuration
st.set_page_config(
    page_title="RAG System - PDF Q&A",
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
    .answer-box {
        background: rgba(30, 136, 229, 0.08);
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-size: 1.1rem;
        line-height: 1.6;
        border: 1px solid rgba(30, 136, 229, 0.15);
    }
    .source-box {
        background: rgba(30, 136, 229, 0.05);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def load_models():
    """Load embedding model and vector store"""
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb
        
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        return model, collection
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def query_huggingface(prompt, hf_token=None):
    """Query HuggingFace API"""
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 400,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', 'No response')
            return str(result)
        else:
            return f"API Error: {response.status_code}. Try adding a HuggingFace token."
    except Exception as e:
        return f"Error: {str(e)}"

def process_pdf(file_path):
    """Extract text from PDF"""
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except:
        pass
    
    try:
        import PyPDF2
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    except:
        pass
    
    return ""

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks

# Initialize session state
if 'initialized' not in st.session_state:
    with st.spinner('🚀 Loading models...'):
        st.session_state.model, st.session_state.collection = load_models()
        st.session_state.initialized = True
        st.session_state.hf_token = ""
        st.session_state.chat_history = []

# Header
st.markdown('<div class="main-header">🤖 RAG System</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">Upload PDFs and ask questions - Powered by AI</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 System")
    
    # HuggingFace Token
    with st.expander("🔑 HuggingFace Token", expanded=not st.session_state.hf_token):
        st.info("Get faster responses with a FREE token from [HuggingFace](https://huggingface.co/settings/tokens)")
        token_input = st.text_input("Token:", type="password", value=st.session_state.hf_token)
        if st.button("Save Token"):
            st.session_state.hf_token = token_input
            st.success("✅ Saved!")
    
    st.markdown("---")
    
    # Document count
    try:
        doc_count = st.session_state.collection.count()
        st.metric("Documents", doc_count)
    except:
        st.metric("Documents", 0)
    
    st.markdown("---")
    
    # PDF Upload
    st.markdown("### 📄 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=['pdf', 'txt'])
    
    if uploaded_file:
        if st.button("🚀 Process", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                try:
                    import tempfile
                    import os
                    
                    # Save temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    # Extract text
                    if uploaded_file.name.endswith('.pdf'):
                        text = process_pdf(tmp_path)
                    else:
                        text = uploaded_file.getvalue().decode('utf-8')
                    
                    if text:
                        # Chunk text
                        chunks = chunk_text(text)
                        
                        # Create embeddings
                        embeddings = st.session_state.model.encode(chunks)
                        
                        # Add to vector store
                        st.session_state.collection.add(
                            documents=chunks,
                            embeddings=embeddings.tolist(),
                            metadatas=[{'source': uploaded_file.name} for _ in chunks],
                            ids=[f"doc_{int(time.time())}_{i}" for i in range(len(chunks))]
                        )
                        
                        st.success(f"✅ Added {len(chunks)} chunks!")
                        os.unlink(tmp_path)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ No text extracted")
                        os.unlink(tmp_path)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    if st.button("🔄 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Main area
st.markdown("### 💬 Ask Questions")

question = st.text_input("Your question:", placeholder="What is this document about?")

if st.button("🚀 Ask", type="primary"):
    if question:
        with st.spinner('🤔 Thinking...'):
            # Search
            query_emb = st.session_state.model.encode([question])
            results = st.session_state.collection.query(
                query_embeddings=query_emb.tolist(),
                n_results=3
            )
            
            # Build context
            context = "\n\n".join(results['documents'][0]) if results['documents'][0] else "No context"
            
            # Generate answer
            prompt = f"""Based on the context below, answer the question. Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""
            
            answer = query_huggingface(prompt, st.session_state.hf_token)
            
            # Save to history
            st.session_state.chat_history.append({
                'q': question,
                'a': answer,
                's': results['documents'][0][:2] if results['documents'] else []
            })

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"**🙋 Q:** {chat['q']}")
        st.markdown(f"""
        <div class="answer-box">
            <strong>🤖 Answer:</strong><br>{chat['a']}
        </div>
        """, unsafe_allow_html=True)
        
        if chat['s']:
            with st.expander(f"📚 Sources ({len(chat['s'])})"):
                for i, src in enumerate(chat['s'], 1):
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>Source {i}:</strong><br>{src[:300]}...
                    </div>
                    """, unsafe_allow_html=True)
        st.markdown("---")
else:
    st.info("""
    👋 **Welcome!**
    
    1. Upload a PDF using the sidebar
    2. Wait for processing
    3. Ask questions about your document!
    
    **No HuggingFace token?** Get one FREE at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 0.9rem;">
    🤖 RAG System | FREE & Open Source | Deployed on Streamlit Cloud ☁️
</div>
""", unsafe_allow_html=True)

