
from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings,
)
from langchain_community.vectorstores import Chroma



extracted_data = load_pdf_file(data_dir="data/")
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

embeddings = download_hugging_face_embeddings()

chroma_persist_directory = "chroma_db"

vectorstore = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory=chroma_persist_directory,
)

# Ensure data is written to disk for reuse by the app.
vectorstore.persist()
