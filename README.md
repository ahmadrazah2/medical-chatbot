Below is a clean, GitHub-ready **README.md** for your **Medical Chatbot (RAG)** project (LangChain + Chroma + Mistral GGUF + multilingual embeddings).

```markdown
# 🏥 Medical Chatbot (RAG) — LangChain + Chroma + Mistral (GGUF)

A **Retrieval-Augmented Generation (RAG)** medical chatbot built with **LangChain**.  
It answers questions using a **medical book** as a knowledge base, retrieves relevant chunks from **ChromaDB**, and generates responses using a **local Mistral 7B Instruct GGUF** model.

---

## ✨ Features

- ✅ **RAG pipeline** (retrieve + generate)
- ✅ **Local LLM** using GGUF model (fast, offline)
- ✅ **Multilingual embeddings** (Korean + English support)
- ✅ **Chroma vector database** with persistence (reuse embeddings without re-indexing)
- ✅ Document chunking with overlap for better retrieval

---

## 🧠 Tech Stack

- **LLM:** `Mistral-7B-Instruct-v0.3.Q6_K.gguf`
- **Embeddings:** `intfloat/multilingual-e5-large`
- **Framework:** LangChain
- **Vector Store:** ChromaDB (persistent)

---

## 📁 Project Structure (example)
system_prompt = (
    "You are a medical assistant for question-answering tasks. "
    "Use ONLY the retrieved context to answer. "
    "If the retrieved context is empty or does not contain the answer, say you don't know. "
    "Answer in English. "
    "Do not invent information outside the provided context. "
    "Keep answers concise (max 3 sentences).\n\n"
    "Context:\n{context}\n"
)


```

medical-chatbot/
│
├── app.py
├── src/
│   ├── helper.py
│   ├── prompt.py
│   └── ...
│
├── data/
│   └── medical_book/   # your medical book files (txt/pdf->txt/etc.)
│
├── chroma_db/          # persisted vector DB (auto-created)
├── models/
│   └── mistral_models/
│       └── 7B-Instruct-v0.3-GGUF/
│           └── Mistral-7B-Instruct-v0.3.Q6_K.gguf
│
├── requirements.txt
└── README.md

````

---

## 📌 Model & Paths

### Local LLM Path (GGUF)

Your project uses this model path:

```python
from pathlib import Path

llm_path = (
    Path.home()
    / "mistral_models"
    / "7B-Instruct-v0.3-GGUF"
    / "Mistral-7B-Instruct-v0.3.Q6_K.gguf"
)
````

### Embedding Model

```python
embedding_model = "intfloat/multilingual-e5-large"
```

---

## 🔎 Document Chunking

Documents are split into chunks before storing in Chroma:

* `chunk_size = 500`
* `chunk_overlap = 20`

Function signature:

```python
extracted_data: List[Document],
chunk_size: int = 500,
chunk_overlap: int = 20,
) -> List[Document]
```

---

## 🗃️ Vector Store (Chroma) + Persistence

Chroma is created from document chunks and persisted to disk:

```python
vectorstore = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory=chroma_persist_directory,
)

# Ensure data is written to disk for reuse by the app.
vectorstore.persist()
```

✅ This means **you only embed once**. Next time, the app can reuse `chroma_db/`.

---

## ⚙️ Installation

### 1) Create environment (recommended)

```bash
conda create -n medbot python=3.10 -y
conda activate medbot
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
python app.py
```

If you are using Flask UI:

* Open: `http://127.0.0.1:5000`

---

## ✅ How it Works (RAG Flow)

1. Load medical book documents
2. Split into chunks (500 chars, overlap 20)
3. Embed chunks with multilingual-e5-large
4. Store embeddings in ChromaDB
5. On user query:

   * Retrieve top relevant chunks
   * Send context + question to local Mistral 7B
   * Return final answer (Korean/English based on user input)

---

## 🧪 Notes / Tips

* **Q6_K** is a strong balance of quality + speed for **16GB RAM** machines.
* If answers feel weak:

  * increase retrieved documents (k)
  * increase chunk size slightly (e.g., 700)
  * improve prompt to force “use context only”

---

## 📜 License

This project is for educational/research use.
(You can add an MIT License if you want.)

---

## 👤 Author

**Ahmad Raza**
AI & Computer Vision Engineer | RAG + LLM Systems

```

If you want, paste your **actual folder names** (`src/`, `data/`, etc.) and I’ll adjust the README to match your repo exactly + add a proper `requirements.txt` section.
```
