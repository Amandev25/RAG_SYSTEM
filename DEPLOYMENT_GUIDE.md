# 🚀 Streamlit Cloud Deployment Guide

This guide will help you deploy your RAG system to Streamlit Cloud for FREE!

## 📋 Prerequisites

1. **GitHub Account** - [Sign up here](https://github.com)
2. **Streamlit Cloud Account** - [Sign up here](https://streamlit.io/cloud)
3. **HuggingFace Account (Optional but Recommended)** - [Sign up here](https://huggingface.co)

---

## 📦 Files to Push to GitHub

### ✅ **INCLUDE These Files/Folders:**

```
✅ web_ui_cloud.py           # Cloud-compatible web UI
✅ requirements_streamlit.txt # Dependencies for Streamlit Cloud
✅ packages.txt              # System packages
✅ src/                      # Source code folder
   ├── document_processor.py
   ├── __init__.py
   └── (other source files)
✅ config_free.yaml          # Configuration
✅ README.md                 # Project description
✅ .gitignore                # Git ignore rules
✅ data/                     # Sample data (optional)
   └── ai_knowledge.txt
```

### ❌ **EXCLUDE These (already in .gitignore):**

```
❌ venv/                     # Virtual environment
❌ __pycache__/              # Python cache
❌ chroma_db/                # Local vector database
❌ my_db/                    # Local database
❌ .env                      # Environment variables
❌ *.log                     # Log files
❌ START_UI.bat              # Local startup scripts
❌ rag_cli.py                # CLI scripts (not needed for cloud)
```

---

## 🔧 Step-by-Step Deployment

### **Step 1: Prepare Your Repository**

1. **Initialize Git** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit - RAG System"
   ```

2. **Create GitHub Repository**:
   - Go to [GitHub](https://github.com/new)
   - Create a new repository (e.g., "rag-system")
   - Don't initialize with README (you already have one)

3. **Push to GitHub**:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/rag-system.git
   git branch -M main
   git push -u origin main
   ```

### **Step 2: Get HuggingFace Token (Optional but Recommended)**

1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Give it a name (e.g., "RAG System")
4. Select "Read" permissions
5. Copy the token (save it somewhere safe!)

### **Step 3: Deploy to Streamlit Cloud**

1. **Go to Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub

2. **Create New App**:
   - Click "New app"
   - Select your repository: `YOUR_USERNAME/rag-system`
   - Branch: `main`
   - Main file path: `web_ui_cloud.py`

3. **Add Secrets (Optional)**:
   - Click "Advanced settings"
   - In "Secrets" section, add:
     ```toml
     HF_TOKEN = "your_huggingface_token_here"
     ```
   - This enables faster LLM responses

4. **Deploy**:
   - Click "Deploy!"
   - Wait 2-5 minutes for deployment

### **Step 4: Test Your App**

1. Once deployed, you'll get a URL like: `https://YOUR_APP.streamlit.app`
2. Upload a PDF using the sidebar
3. Ask questions about it!

---

## 🎯 Important Notes

### **Differences from Local Version:**

| Feature | Local (Ollama) | Cloud (HuggingFace) |
|---------|----------------|---------------------|
| **LLM** | Ollama (phi3:mini) | HuggingFace (Mistral-7B) |
| **Speed** | Very Fast | Slower (API calls) |
| **Privacy** | 100% Private | Data sent to HuggingFace |
| **Cost** | FREE | FREE (with rate limits) |
| **Setup** | Requires Ollama | No setup needed |

### **Rate Limits:**

- **Without Token**: ~30 requests/hour
- **With Token**: ~300 requests/hour (FREE tier)

### **Storage:**

- Vector database persists between sessions
- Uploaded documents remain in the database
- Free tier: 1GB storage

---

## 🔍 Troubleshooting

### **"Module not found" errors:**
- Check that `requirements_streamlit.txt` is in your repo
- Streamlit Cloud looks for `requirements.txt` by default
- Rename `requirements_streamlit.txt` to `requirements.txt`

### **"Model loading failed":**
- First load takes 2-3 minutes (downloading models)
- Refresh the page after a few minutes

### **"API timeout":**
- HuggingFace model might be loading
- Wait 20-30 seconds and try again
- Add your HF token for priority access

### **"Out of memory":**
- Reduce chunk size in document processing
- Process smaller PDFs
- Streamlit Cloud has 1GB RAM limit

---

## 🚀 Quick Deploy Commands

```bash
# 1. Add all files
git add .

# 2. Commit changes
git commit -m "Deploy RAG system to cloud"

# 3. Push to GitHub
git push origin main

# Streamlit Cloud will auto-deploy!
```

---

## 📊 Performance Tips

1. **Use HuggingFace Token** - Faster responses
2. **Keep PDFs under 10MB** - Faster processing
3. **Limit to 3-5 documents** - Better performance
4. **Clear old documents** - Saves storage

---

## 🆓 Cost Breakdown

| Component | Cost |
|-----------|------|
| Streamlit Cloud Hosting | **FREE** |
| HuggingFace Inference API | **FREE** |
| ChromaDB Vector Storage | **FREE** |
| Sentence Transformers | **FREE** |
| **TOTAL** | **$0.00/month** |

---

## ✅ Checklist Before Deploying

- [ ] `.gitignore` file created
- [ ] `web_ui_cloud.py` exists
- [ ] `requirements_streamlit.txt` renamed to `requirements.txt`
- [ ] `packages.txt` exists
- [ ] `src/` folder included
- [ ] Pushed to GitHub
- [ ] HuggingFace token ready (optional)
- [ ] Streamlit Cloud account created

---

## 🎉 You're Ready!

Your RAG system will be live at: `https://YOUR_APP.streamlit.app`

Share it with friends, colleagues, or the world! 🌍

---

**Need Help?** 
- Streamlit Community: [discuss.streamlit.io](https://discuss.streamlit.io)
- HuggingFace: [discuss.huggingface.co](https://discuss.huggingface.co)

