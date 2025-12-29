# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms import LlamaCpp
# from typing import List
# from langchain.schema import Document
# from pathlib import Path
# import multiprocessing


# #Extract Data From the PDF File
# def load_pdf_file(data):
#     loader= DirectoryLoader(data,
#                             glob="*.pdf",
#                             loader_cls=PyPDFLoader)

#     documents=loader.load()

#     return documents



# def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
#     """
#     Given a list of Document objects, return a new list of Document objects
#     containing only 'source' in metadata and the original page_content.
#     """
#     minimal_docs: List[Document] = []
#     for doc in docs:
#         src = doc.metadata.get("source")
#         minimal_docs.append(
#             Document(
#                 page_content=doc.page_content,
#                 metadata={"source": src}
#             )
#         )
#     return minimal_docs



# #Split the Data into Text Chunks
# def text_split(extracted_data):
#     text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
#     text_chunks=text_splitter.split_documents(extracted_data)
#     return text_chunks



# #Download the Embeddings from HuggingFace 
# def download_hugging_face_embeddings():
#     # Strong multilingual embeddings for better English/Korean recall
#     embeddings = HuggingFaceEmbeddings(
#         model_name="intfloat/multilingual-e5-large"
#     )
#     return embeddings


# def load_local_llm(
#     model_path: Path | None = None,
#     n_ctx: int = 2048,
#     n_threads: int | None = None,
#     n_gpu_layers: int = 0,
# ):
#     """
#     Load a local GGUF model via LlamaCpp. Defaults target the shipped mistral model path.
#     Adjust n_gpu_layers>0 if you have GPU offload enabled; leave 0 for CPU-only.
#     """
#     if model_path is None:
#         model_path = (
#             Path.home()
#             / "mistral_models"
#             / "7B-Instruct-v0.3-GGUF"
#             / "Mistral-7B-Instruct-v0.3.Q6_K.gguf"
#         )

#     if n_threads is None:
#         n_threads = max(multiprocessing.cpu_count() - 1, 1)

#     return LlamaCpp(
#         model_path=str(model_path),
#         n_ctx=n_ctx,
#         n_threads=n_threads,
#         n_gpu_layers=n_gpu_layers,
#         temperature=0.0,
#         max_tokens=256,
#         stop=["</s>"],
#         verbose=False,
#     )
# src/helper.py
# from __future__ import annotations

import multiprocessing
import re
from pathlib import Path
from typing import List, Optional

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.schema import Document


# =========================
# 1) PDF LOADING
# =========================
def load_pdf_file(data_dir: str) -> List[Document]:
    """
    Load all PDFs from a directory (non-recursive by default, depends on DirectoryLoader).
    Example: load_pdf_file("data/pdfs")
    """
    loader = DirectoryLoader(
        data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Keep only minimal metadata (source) to reduce storage noise.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src},
            )
        )
    return minimal_docs


# =========================
# 2) SPLITTING (CHUNKING)
# =========================
def text_split(
    extracted_data: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 20,
) -> List[Document]:
    """
    Split loaded documents into chunks for embedding + retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(extracted_data)


# =========================
# 3) MULTILINGUAL E5 EMBEDDINGS (recommended)
# =========================
class E5Embeddings(HuggingFaceEmbeddings):
    """
    E5 models work best with:
      - "passage: ..." for documents
      - "query: ..." for queries
    """

    def embed_documents(self, texts):
        texts = [f"passage: {t}" for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")


def download_hugging_face_embeddings():
    """
    Strong multilingual embeddings for English/Korean cross-lingual retrieval.
    """
    return E5Embeddings(
        model_name="intfloat/multilingual-e5-large",
        encode_kwargs={"normalize_embeddings": True},
    )


# =========================
# 4) LOCAL LLM (LLamaCpp)
# =========================
def load_local_llm(
    model_path: Optional[Path] = None,
    n_ctx: int = 1024,
    n_threads: Optional[int] = None,
    n_gpu_layers: int = 99,
    temperature: float = 0.0,
    max_tokens: int = 128,
):
    """
    Load a local GGUF model via LlamaCpp.

    - n_gpu_layers>0 if you have GPU offload enabled
    - leave n_gpu_layers=0 for CPU-only
    """
    if model_path is None:
        model_path = (
            Path.home()
            / "mistral_models"
            / "7B-Instruct-v0.3-GGUF"
            / "Mistral-7B-Instruct-v0.3.Q6_K.gguf"
        )

    if n_threads is None:
        n_threads = max(multiprocessing.cpu_count() - 1, 1)

    return LlamaCpp(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["</s>"],
        verbose=False,
    )


# =========================
# 5) TEXT / LANGUAGE HELPERS
# (If you don't want src/utils.py, use these from helper.py)
# =========================
def clean_msg(text: str) -> str:
    """
    Removes trailing timestamps like '13:3' or '13:04' that your UI might append.
    """
    text = text.strip()
    text = re.sub(r"\s*\d{1,2}:\d{1,2}\s*$", "", text)
    return text.strip()

