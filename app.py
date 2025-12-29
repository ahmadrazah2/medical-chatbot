# from flask import Flask, render_template, jsonify, request
# from src.helper import download_hugging_face_embeddings, load_local_llm
# from langchain_community.vectorstores import Chroma
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import PromptTemplate
# from dotenv import load_dotenv
# from src.prompt import *
# import os
# import re


# app = Flask(__name__)


# load_dotenv()

# embeddings = download_hugging_face_embeddings()

# chroma_persist_directory = "chroma_db"
# vectorstore = Chroma(
#     persist_directory=chroma_persist_directory,
#     embedding_function=embeddings,
# )


# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# llm = load_local_llm(n_ctx=2048, n_gpu_layers=0)

# prompt = PromptTemplate.from_template(
#     system_prompt + "\n\nQuestion: {input}"
# )

# question_answer_chain = create_stuff_documents_chain(llm, prompt)


# def is_korean(text: str) -> bool:
#     return re.search(r"[\uac00-\ud7a3]", text) is not None


# def ko_to_en_query(llm_model, ko_text: str) -> str:
#     translate_prompt = (
#         "Translate this Korean medical question to English for document retrieval. "
#         "Keep meaning precise. Output ONLY the English question.\n\n"
#         f"Korean: {ko_text}\nEnglish:"
#     )
#     translation = llm_model.invoke(translate_prompt)
#     if isinstance(translation, str):
#         text = translation.strip()
#     else:
#         text = getattr(translation, "content", str(translation)).strip()
#     return text or ko_text.strip()



# @app.route("/")
# def index():
#     return render_template('chat.html')



# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"].strip()
#     if not msg:
#         return "I don't know."

#     if is_korean(msg):
#         retrieval_query = ko_to_en_query(llm, msg)
#     else:
#         retrieval_query = msg

#     docs = retriever.invoke(retrieval_query)
#     print(f"[DEBUG] retrieval_query={retrieval_query!r} docs_found={len(docs)}")

#     if not docs:
#         return "모르겠습니다." if is_korean(msg) else "I don't know."

#     result = question_answer_chain.invoke(
#         {
#             "input": msg,
#             "context": docs,
#         }
#     )

#     answer = result if isinstance(result, str) else result.get("output_text", str(result))
#     print("Response : ", answer)
#     return str(answer)



# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=False)
from flask import Flask, render_template, request
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
import os

# Fix tokenizers parallelism warning in forked processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# M2 Silicon optimization
os.environ["GGML_METAL"] = "1"  # Enable Metal GPU acceleration for M2

from src.helper import (
    download_hugging_face_embeddings,
    load_local_llm,
    clean_msg,
)
from src.prompt import system_prompt

load_dotenv()

app = Flask(__name__)

# Load local models only (Mistral + E5)
embeddings = download_hugging_face_embeddings()
llm = load_local_llm(n_gpu_layers=99)

# Vector DB
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 4, "score_threshold": 0.35},
)

prompt = PromptTemplate.from_template(
    system_prompt + "\n\nQuestion: {input}"
)

qa_chain = create_stuff_documents_chain(llm, prompt)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = clean_msg(request.form["msg"])
    docs = retriever.invoke(msg)

    if not docs:
        return "I don't know."

    result = qa_chain.invoke({"input": msg, "context": docs})

    answer_en = result if isinstance(result, str) else result["output_text"]
    # Final cleanup of timestamps and extra whitespace
    return clean_msg(answer_en)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)
