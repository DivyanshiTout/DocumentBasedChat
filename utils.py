import os
import logging
from langchain_community.document_loaders import (
    UnstructuredPDFLoader, UnstructuredWordDocumentLoader, TextLoader
)
from huggingface_hub import hf_hub_download

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.llms import LlamaCpp

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
VECTOR_COLLECTION_NAME = "doc_qa"
# ✅ Download the model from Hugging Face (once on startup)
MODEL_PATH = hf_hub_download(
    repo_id="abhishek-sen/document-qa",  # e.g. abhishek-sen/document-qa
    filename="llama3.2.Q4_K_M.gguf",
    token=os.getenv("HF_TOKEN")  # Add this in Render environment settings
)

# ✅ Load the model
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.7,
    max_tokens=512,
    n_ctx=2048
)


def extract_model_names(models_info):
    try:
        if hasattr(models_info, "models"):
            return tuple(model.model for model in models_info.models)
        return tuple()
    except Exception as e:
        logger.error(f"Error extracting model names: {e}")
        return tuple()


def handle_document_upload(file_category_pairs):
    all_chunks = []

    for path, category, display_name in file_category_pairs:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            loader = UnstructuredPDFLoader(path)
        elif ext == ".docx":
            loader = UnstructuredWordDocumentLoader(path)
        elif ext == ".txt":
            loader = TextLoader(path)
        else:
            continue

        data = loader.load()
        for doc in data:
            doc.metadata["category"] = category
            doc.metadata["display_name"] = display_name

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(data)

        for chunk in chunks:
            chunk.metadata["category"] = category
            chunk.metadata["display_name"] = display_name

        all_chunks.extend(chunks)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    existing_collections = [c.name for c in client.get_collections().collections]
    if VECTOR_COLLECTION_NAME not in existing_collections:
        client.recreate_collection(
            collection_name=VECTOR_COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    vector_db = Qdrant(
        client=client,
        collection_name=VECTOR_COLLECTION_NAME,
        embeddings=embeddings,
    )

    vector_db.add_documents(all_chunks)
    return vector_db


def detect_message_type(question, model_name="llama3"):
    llm_check = ChatOllama(model=model_name, streaming=False, temperature=0)
    check_prompt = f"""
Classify the following message as one of the following types:
- greeting
- Expression of Gratitude
- other

Only respond with one word: greeting, gratitude, or other.

Message:
{question}
"""
    response = llm_check.invoke(check_prompt).content.strip().lower()
    return response if response in ["greeting", "gratitude"] else "other"




# def process_question(question, vector_db, selected_model, role):
#     llm = ChatOllama(model=selected_model, streaming=True, max_tokens=100)

#     # Define retriever with custom filter function based on role
#     if role == "employee":
#         filter_fn = lambda doc: doc.metadata.get("category") == "private"
#     else:
#         filter_fn = lambda doc: doc.metadata.get("category") == "public"

#     retriever = vector_db.as_retriever(
#         search_kwargs={"k": 3},
#         filter=filter_fn
#     )
#     results_with_scores = vector_db.similarity_search_with_score(question, k=3)
#     similarity_threshold = 1.1
#     retrieved_doc = [doc for doc, score in results_with_scores if score > similarity_threshold]

#     # Manually retrieve documents first
#     retrieved_docs = retriever.invoke(question)

#     # if is_greeting_via_llm(question, selected_model):
#     #     llm_greet = ChatOllama(model=selected_model, streaming=True, temperature=0.7)
#     #     return llm_greet.invoke(question).content  #

#     message_type = detect_message_type(question, selected_model)
#     print(message_type)
#     if message_type in ["greeting", "gratitude"]:
#         llm = ChatOllama(model=selected_model, streaming=True, temperature=0.7)
#         return llm.invoke(question).content
#     else:
#         if retrieved_doc:
#             return "I am designed to assist with XYZ organization-related queries only. Please ask something related."

#     if role != "employee":

#         # Check the categories of matched docs
#         public_found = any(doc.metadata.get('category') == "public" for doc in retrieved_docs)
#         if public_found:
#             context = "\n\n".join([doc.page_content for doc in retrieved_docs])

#             RAG_TEMPLATE = """
#             Answer the question using ONLY the following context:
#             - Be brief and to the point.
#             - Limit the answer to 2–3 sentences max.
#             - If the question is a greeting, respond with a friendly and short greeting.

#             {context}

#             Question: {question}
#             """

#             prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
#             context_runnable = RunnableLambda(lambda _: context)

#             chain = (
#                 {"context": context_runnable, "question": RunnablePassthrough()}
#                 | prompt
#                 | llm
#                 | StrOutputParser()
#             )

#             return chain.invoke(question)
#         else:
#             return "You are not an authenticated person. Please connect with the HR team."
#     else:
#         RAG_TEMPLATE = """
#         Answer the question using ONLY the following context:
#         - Be brief and to the point.
#         - Limit the answer to 2–3 sentences max.
#         - If the question is a greeting, respond with a friendly and short greeting.

#         {context}

#         Question: {question}
#         """

#         prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

#         chain = (
#             {"context": retriever, "question": RunnablePassthrough()}
#             | prompt
#             | llm
#             | StrOutputParser()
#         )

#             # Invoke the chain
#         answer = chain.invoke(question)

#         return answer
    



def process_question(question, vector_db, role):
    # Role-based document filter
    filter_fn = (
        lambda doc: doc.metadata.get("category") == "private"
        if role == "employee"
        else lambda doc: doc.metadata.get("category") == "public"
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": 3}, filter=filter_fn)

    # Similarity check
    results_with_scores = vector_db.similarity_search_with_score(question, k=3)
    retrieved_doc = [doc for doc, score in results_with_scores if score > 1.1]
    retrieved_docs = retriever.invoke(question)

    # Optional: detect greeting/gratitude
    message_type = detect_message_type(question)  # implement this function
    if message_type in ["greeting", "gratitude"]:
        return llm.invoke(question)

    # Reject unrelated questions
    if retrieved_doc:
        return "I am designed to assist with XYZ organization-related queries only. Please ask something related."

    # Define prompt
    RAG_TEMPLATE = """
    Answer the question using ONLY the following context:
    - Be brief and to the point.
    - Limit the answer to 2–3 sentences max.
    - If the question is a greeting, respond with a friendly and short greeting.

    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    # Public user logic
    if role != "employee":
        public_found = any(doc.metadata.get("category") == "public" for doc in retrieved_docs)
        if not public_found:
            return "You are not an authenticated person. Please connect with the HR team."

        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        context_runnable = RunnableLambda(lambda _: context)
        chain = (
            {"context": context_runnable, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain.invoke(question)

    # Employee logic
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(question)

