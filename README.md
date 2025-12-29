
# 🏥 Medical Chatbot using RAG (LangChain + Mistral)

A **Retrieval-Augmented Generation (RAG)** based **medical chatbot** that answers medical questions using a **medical book as knowledge base**.  
The system uses **LangChain**, **ChromaDB**, **multilingual embeddings**, and a **local Mistral 7B Instruct GGUF model** for offline, fast, and reliable inference.

---

## 🚀 Features

- 🔍 Retrieval-Augmented Generation (RAG)
- 🧠 Local LLM inference (no paid API required)
- 📚 Medical book–based knowledge retrieval
- 💾 Persistent vector database (ChromaDB)
- ⚡ Optimized GGUF model (Q6_K)

---

## 🧠 Technology Stack

| Component | Tool |
|---------|------|
| LLM | Mistral-7B-Instruct (GGUF) |
| Framework | LangChain |
| Embeddings | intfloat/multilingual-e5-large |
| Vector Store | ChromaDB |
| Language | Python |
| Deployment | Local (Offline) |

---

## 📁 Project Structure

```

medical-chatbot-rag/
│
├── app.py
├── src/
│   ├── helper.py
│   ├── prompt.py
│   └── **init**.py
│
├── data/
│   └── medical_book/      # Medical book text files
│
├── chroma_db/             # Persisted vector database
│
├── models/
│   └── mistral_models/
│       └── 7B-Instruct-v0.3-GGUF/
│           └── Mistral-7B-Instruct-v0.3.Q6_K.gguf
│
├── requirements.txt
└── README.md

````

---

## 🤖 Local LLM Configuration

The chatbot uses a **local GGUF model**:

```python
from pathlib import Path

llm_path = (
    Path.home()
    / "mistral_models"
    / "7B-Instruct-v0.3-GGUF"
    / "Mistral-7B-Instruct-v0.3.Q6_K.gguf"
)
````

**Recommended RAM:** 16 GB
**Context length:** 4096 tokens

---

## 🔎 Embeddings & Chunking

### Embedding Model

```python
intfloat/multilingual-e5-large
```

### Text Chunking

* `chunk_size = 500`
* `chunk_overlap = 20`

```python
extracted_data: List[Document],
chunk_size: int = 500,
chunk_overlap: int = 20,
```

---

## 🗃️ Vector Store (ChromaDB)

Documents are embedded and stored persistently:

```python
vectorstore = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory=chroma_persist_directory,
)

vectorstore.persist()
```

✔ Embeddings are generated **once** and reused on every run.

---

## ⚙️ Installation

### 1️⃣ Create Environment

```bash
conda create -n medicalbot python=3.10 -y
conda activate medicalbot
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Chatbot

```bash
python app.py
```

If using Flask UI:

```
http://127.0.0.1:5000
```

---

## 🔄 RAG Pipeline Workflow

1. Load medical book documents
2. Split text into overlapping chunks
3. Generate multilingual embeddings
4. Store vectors in ChromaDB
5. Retrieve relevant context for user query
6. Generate final answer using Mistral 7B

---

## 🧪 Tips for Better Results

* Increase retrieval `k` for deeper context
* Improve prompt to force **context-only answers**
* Use higher chunk size for long explanations
* Use Korean prompt template for Korean queries

---

## 📌 Limitations

* Not a replacement for professional medical advice
* Depends on quality of medical book data
* Local inference speed depends on hardware

---

## 📜 License

This project is intended for **educational and research purposes**.

---

## 👤 Author

**Ahmad Raza**
AI & Computer Vision Engineer
Research Focus: RAG Systems, LLMs, Medical AI

🔗 GitHub: [https://github.com/ahmadrazah2](https://github.com/ahmadrazah2)

```


Just tell me 👍
```
