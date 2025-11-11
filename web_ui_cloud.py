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
    """Initialize the RAG system components with PERSISTENT storage"""
    from sentence_transformers import SentenceTransformer
    import chromadb
    import os
    
    # Initialize embedding model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # PERSIST ChromaDB to disk (survives cloud restarts!)
    persist_dir = "./chroma_db_cloud"
    os.makedirs(persist_dir, exist_ok=True)
    
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name="rag_documents_v2",  # New version with compressed chunks
        metadata={"hnsw:space": "cosine"}
    )
    
    return model, collection

def generate_answer_from_context(question, context_docs):
    """Generate answer directly from retrieved documents (no LLM needed!)"""
    if not context_docs:
        return "I couldn't find any relevant information in your documents to answer this question."
    
    import re
    from collections import Counter
    
    # Combine and clean up the context
    full_context = " ".join(context_docs)
    
    # Fix common PDF extraction issues
    # Add spaces before capital letters that follow lowercase letters
    full_context = re.sub(r'([a-z])([A-Z])', r'\1 \2', full_context)
    # Remove excessive whitespace
    full_context = re.sub(r'\s+', ' ', full_context).strip()
    
    # Extract sentences
    sentences = re.split(r'(?<=[.!?])\s+', full_context)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]  # Filter short fragments
    
    # Remove duplicates while preserving order
    seen = set()
    unique_sentences = []
    for sentence in sentences:
        sentence_normalized = sentence.lower()[:100]  # Compare first 100 chars
        if sentence_normalized not in seen:
            seen.add(sentence_normalized)
            unique_sentences.append(sentence)
    
    # Find most relevant sentences based on question keywords
    question_words = set(question.lower().split())
    question_words = {w for w in question_words if len(w) > 3}  # Only meaningful words
    
    # Score sentences
    scored_sentences = []
    for sentence in unique_sentences:
        sentence_lower = sentence.lower()
        # Count keyword matches
        matches = sum(1 for word in question_words if word in sentence_lower)
        # Prefer sentences at the beginning (likely to be definitions/introductions)
        position_score = 1.0 / (unique_sentences.index(sentence) + 1)
        # Prefer sentences with moderate length (not too short, not too long)
        length_score = 1.0 if 50 < len(sentence) < 300 else 0.5
        
        total_score = matches + (position_score * 0.3) + (length_score * 0.2)
        
        if total_score > 0:
            scored_sentences.append((total_score, sentence))
    
    # Sort by relevance
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    
    # Take top 3-4 most relevant sentences
    top_sentences = [s[1] for s in scored_sentences[:4]]
    
    if top_sentences:
        # Create a structured summary
        answer_parts = []
        
        # Try to identify if first sentence is a definition
        first_sentence = top_sentences[0]
        if any(word in first_sentence.lower() for word in ['is', 'refers to', 'means', 'defined as']):
            answer_parts.append(f"**Definition:** {first_sentence}")
            remaining = top_sentences[1:]
        else:
            remaining = top_sentences
        
        # Add remaining sentences as key points
        if remaining:
            if answer_parts:
                answer_parts.append(f"\n**Key Points:**")
            for i, sentence in enumerate(remaining, 1):
                answer_parts.append(f"{i}. {sentence}")
        
        answer_text = "\n\n".join(answer_parts)
    else:
        # Fallback: show beginning of context
        answer_text = full_context[:800] + "..."
    
    answer = f"""Based on your documents:

{answer_text}

---

⚠️ **Limited Explanation:** Without an AI model, I can only extract and organize text from your documents.

🎯 **For Real AI Explanations:** Add a FREE HuggingFace token (takes 2 min):
   • Get token: https://huggingface.co/settings/tokens
   • Add to Streamlit Secrets: `HF_TOKEN = "your_token"`
   • Get AI-generated summaries, better answers, and natural explanations!"""
    
    return answer

def query_huggingface(prompt, hf_token=None):
    """Query HuggingFace API with multiple model fallbacks"""
    
    # If no token, return None to trigger local answer generation
    if not hf_token or hf_token == "":
        return None
    
    # Try multiple models in order of preference
    models = [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "microsoft/Phi-3-mini-4k-instruct",
        "google/flan-t5-large"
    ]
    
    headers = {"Authorization": f"Bearer {hf_token}"}
    
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
    
    # If all models fail, provide helpful message
    return """⚠️ AI models are currently unavailable. To get full AI-generated answers:

📌 Add a FREE HuggingFace token (takes 2 minutes):
   1. Get token: https://huggingface.co/settings/tokens
   2. In Streamlit Cloud: Settings → Secrets, add:
      HF_TOKEN = "your_token_here"
   3. Reboot the app

💡 With a token you get:
   ✅ Priority access to AI models
   ✅ No rate limits  
   ✅ Faster responses
   ✅ 100% FREE!"""

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
        try:
            # Try to read from Streamlit secrets (both syntaxes)
            if "HF_TOKEN" in st.secrets:
                st.session_state.hf_token = st.secrets["HF_TOKEN"]
            else:
                st.session_state.hf_token = ""
        except Exception as e:
            st.session_state.hf_token = ""
    
    # Show token status with DEBUG INFO
    if st.session_state.hf_token:
        st.success("✅ HuggingFace token configured!")
        # DEBUG: Show token is actually loaded
        token_preview = st.session_state.hf_token
        if len(token_preview) > 8:
            masked = f"{token_preview[:4]}...{token_preview[-4:]}"
        else:
            masked = "***"
        st.info(f"🔍 **DEBUG:** Token detected: `{masked}`")
        st.caption(f"Token length: {len(st.session_state.hf_token)} characters")
    else:
        st.error("❌ NO TOKEN DETECTED!")
        with st.expander("🔑 HuggingFace Token Required", expanded=True):
            st.warning("⚠️ Token not found in secrets. Add it for AI models to work!")
            st.info("""
            **How to add token:**
            1. Get token: https://huggingface.co/settings/tokens
            2. In Streamlit Cloud: ⚙️ Settings → Secrets
            3. Add this line:
               ```
               HF_TOKEN = "your_token_here"
               ```
            4. Save & Reboot app
            
            **Or enter temporarily below (won't persist):**
            """)
            hf_input = st.text_input("Token:", type="password", key="hf_input")
            if st.button("Save Token (This Session Only)"):
                st.session_state.hf_token = hf_input
                st.success("✅ Token saved for this session!")
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
        
        # DEBUG: Show retrieved chunks
        if results['documents'][0]:
            with st.expander("🔍 Debug: Retrieved Chunks", expanded=False):
                for i, chunk in enumerate(results['documents'][0], 1):
                    st.text_area(f"Chunk {i}", chunk, height=100)
        
        # COMPRESS chunks to 1-2 sentences each (forces abstraction)
        compressed_chunks = []
        if results['documents'][0]:
            import re
            for i, chunk in enumerate(results['documents'][0], 1):
                # Clean up text
                clean_chunk = re.sub(r'([a-z])([A-Z])', r'\1 \2', chunk)
                clean_chunk = re.sub(r'\s+', ' ', clean_chunk).strip()
                # Extract first 2 meaningful sentences
                sentences = re.split(r'[.!?]+\s+', clean_chunk)
                compressed = '. '.join(sentences[:2])[:300]
                # Add chunk ID for citation
                compressed_chunks.append(f"[Chunk {i}]: {compressed}")
        
        # Build compressed context with chunk IDs
        context = "\n\n".join(compressed_chunks) if compressed_chunks else "No context available"
        
        # Try to generate answer with LLM first, fallback to local RAG
        answer = None
        use_local_rag = True  # Default to local RAG
        
        if st.session_state.hf_token:
            # ULTRA-STRICT PARAPHRASE PROMPT (8-char verbatim limit)
            prompt = f"""SYSTEM:

You are a precise assistant. Using ONLY the supplied short summaries (not raw document text), answer the user's question in your OWN words.

- Provide a concise 2-4 sentence answer for definitions; up to 6 sentences for explanations.
- Do NOT copy more than 8 consecutive characters verbatim from the original documents.
- If you include exact wording, use quotes and include the page/chunk id next to it.
- Add a one-line "Sources: ..." at the end listing chunk IDs or page numbers you used.

Context Summaries:
{context}

Question: {question}

Answer:"""
            
            answer = query_huggingface(prompt, st.session_state.hf_token)
            
            # DEBUG: Show what the LLM returned
            st.info(f"🔍 **DEBUG:** LLM Response Type: {type(answer).__name__}, Length: {len(str(answer)) if answer else 0}")
            if answer:
                st.caption(f"First 100 chars: {str(answer)[:100]}...")
            
            # POST-CHECK: Detect long exact matches (verbatim copying)
            if answer and not (answer.startswith("⚠️") or answer.startswith("❌") or "unavailable" in answer.lower()):
                # Check for verbatim copying
                import difflib
                for chunk in (results['documents'][0] if results['documents'] else []):
                    # Find longest common substring
                    seq = difflib.SequenceMatcher(None, answer.lower(), chunk.lower())
                    match = seq.find_longest_match(0, len(answer), 0, len(chunk))
                    if match.size > 50:  # If >50 chars copied verbatim (stricter!)
                        st.warning("⚠️ Detected verbatim copying. Regenerating with stricter prompt...")
                        # Re-prompt with ULTRA-strict instructions
                        prompt = f"""CRITICAL SYSTEM INSTRUCTION:

You FAILED the previous attempt by copying text verbatim. Try again.

RULES:
- Write ONLY 2-3 SHORT sentences
- Use COMPLETELY DIFFERENT WORDS than the context
- MAXIMUM 8 consecutive characters from source
- Explain as if to a 5-year-old who cannot see the context

Context: {context}
Question: {question}

Your simplified explanation:"""
                        answer = query_huggingface(prompt, st.session_state.hf_token)
                        break
                
                use_local_rag = False
        
        # Use local RAG-based answer if no valid LLM response
        if use_local_rag:
            answer = generate_answer_from_context(question, results['documents'][0] if results['documents'] else [])
            generation_method = "🔧 Local Extraction (No LLM)"
        else:
            generation_method = "🤖 AI LLM Generated"
        
        # Add to chat history with generation method
        st.session_state.chat_history.append({
            'question': question,
            'answer': answer,
            'sources': results['documents'][0][:3] if results['documents'] else [],
            'method': generation_method
        })

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    
    for chat in reversed(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**🙋 You asked:** {chat['question']}")
            
            # Show generation method for debugging
            method = chat.get('method', '❓ Unknown')
            st.caption(f"**Generation Method:** {method}")
            
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

