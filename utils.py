import os
import shutil
from langchain_community.document_loaders import (
    UnstructuredPDFLoader, UnstructuredWordDocumentLoader, TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda


import logging
logger = logging.getLogger(__name__)

PERSIST_DIRECTORY = os.path.join("data", "vectors")

def extract_model_names(models_info):
    try:
        if hasattr(models_info, "models"):
            return tuple(model.model for model in models_info.models)
        return tuple()
    except Exception as e:
        logger.error(f"Error extracting model names: {e}")
        return tuple()

# def handle_document_upload(paths):
#     all_chunks = []
#     for path in paths:
#         ext = os.path.splitext(path)[1].lower()
#         if ext == ".pdf":
#             loader = UnstructuredPDFLoader(path)
#         elif ext == ".docx":
#             loader = UnstructuredWordDocumentLoader(path)
#         elif ext == ".txt":
#             loader = TextLoader(path)
#         else:
#             continue

#         data = loader.load()
#         splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#         chunks = splitter.split_documents(data)
#         all_chunks.extend(chunks)

#     embeddings = OllamaEmbeddings(model="nomic-embed-text")
#     vector_db = Chroma.from_documents(
#         documents=all_chunks,
#         embedding=embeddings,
#         persist_directory=PERSIST_DIRECTORY,
#         collection_name="multi_file_rag"
#     )
#     return vector_db


# def process_question(question, vector_db, selected_model):
#     llm = ChatOllama(model=selected_model, streaming=True,max_tokens=100)

#     QUERY_PROMPT = PromptTemplate(
#         input_variables=["question"],
#         template="""You are an AI language model assistant. Your task is to generate 2
#         different versions of the given user question to retrieve relevant documents from
#         a vector database. Original question: {question}"""
#     )
#     retriever = vector_db.as_retriever(search_kwargs={"k": 3})

#     # retriever = MultiQueryRetriever.from_llm(
#     #     vector_db.as_retriever(search_kwargs={"k": 3}),
#     #     llm,
#     #     prompt=QUERY_PROMPT
#     # )

#     # RAG_TEMPLATE = """Answer the question based ONLY on the following context:
#     # If the question is a general greeting message respond with a warm and friendly greeting message instead of using the context.
#     # {context}
#     # Question: {question}
#     # """

#     RAG_TEMPLATE = """
#     Answer the question using ONLY the following context:
#     - Be brief and to the point.
#     - Limit the answer to 2–3 sentences max.
#     - If the question is a greeting, respond with a friendly and short greeting.

#     {context}

#     Question: {question}
#     """

#     prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

#     chain = (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     return chain.invoke(question)



# def handle_document_upload(paths, user_role):
#     all_chunks = []
#     category = "private" if user_role == "employee" else "public"

#     for path in paths:
#         ext = os.path.splitext(path)[1].lower()
#         if ext == ".pdf":
#             loader = UnstructuredPDFLoader(path)
#         elif ext == ".docx":
#             loader = UnstructuredWordDocumentLoader(path)
#         elif ext == ".txt":
#             loader = TextLoader(path)
#         else:
#             continue

#         data = loader.load()

#         # Assign metadata to each document before splitting
#         for doc in data:
#             doc.metadata["category"] = category

#         splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#         chunks = splitter.split_documents(data)

#         # Assign metadata to each chunk after splitting (in case splitter resets metadata)
#         for chunk in chunks:
#             chunk.metadata["category"] = category

#         all_chunks.extend(chunks)

#     embeddings = OllamaEmbeddings(model="nomic-embed-text")
#     vector_db = Chroma.from_documents(
#         documents=all_chunks,
#         embedding=embeddings,
#         persist_directory=PERSIST_DIRECTORY,
#         collection_name="multi_file_rag"
#     )
#     return vector_db





def handle_document_upload(file_category_pairs):
    all_chunks = []

    for path, category,display_name in file_category_pairs:
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
    vector_db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name="multi_file_rag"
    )
    return vector_db



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
#         # Manual retrieval before passing to LLM
#     retrieved_docs = retriever.invoke(question)

#     if role != "employee":
#         for doc in retrieved_docs:
#             if doc.metadata.get('category') == "private":
#                 return "You are not authorized to access this information."
#             else:
#                 # No private document found — safe to proceed
#                 context = "\n\n".join([doc.page_content for doc in retrieved_docs])

#                 RAG_TEMPLATE = """
#                 Answer the question using ONLY the following context:
#                 - Be brief and to the point.
#                 - Limit the answer to 2–3 sentences max.
#                 - If the question is a greeting, respond with a friendly and short greeting.

#                 {context}

#                 Question: {question}
#                 """

#                 prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
#                 context_runnable = RunnableLambda(lambda _: context)


#                 chain = (
#                     {"context": context_runnable, "question": RunnablePassthrough()}
#                     | prompt
#                     | llm
#                     | StrOutputParser()
#                 )

#                 return chain.invoke(question)
    
#     RAG_TEMPLATE = """
#     Answer the question using ONLY the following context:
#     - Be brief and to the point.
#     - Limit the answer to 2–3 sentences max.
#     - If the question is a greeting, respond with a friendly and short greeting.

#     {context}

#     Question: {question}
#     """

#     prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

#     chain = (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#         # Invoke the chain
#     answer = chain.invoke(question)

#     return answer


# ✅ LLM-based greeting check
def is_greeting_via_llm(question, model_name="llama3"):
    llm_check = ChatOllama(model=model_name, streaming=False, temperature=0)
    check_prompt = f"Is the following message a greeting? Only respond with 'yes' or 'no':\n\n{question}"
    
    # Get only the content (text) from the response
    response = llm_check.invoke(check_prompt).content.strip().lower()
    return response.startswith("yes")



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





def process_question(question, vector_db, selected_model, role):
    llm = ChatOllama(model=selected_model, streaming=True, max_tokens=100)

    # Define retriever with custom filter function based on role
    if role == "employee":
        filter_fn = lambda doc: doc.metadata.get("category") == "private"
    else:
        filter_fn = lambda doc: doc.metadata.get("category") == "public"

    retriever = vector_db.as_retriever(
        search_kwargs={"k": 3},
        filter=filter_fn
    )
    results_with_scores = vector_db.similarity_search_with_score(question, k=3)
    similarity_threshold = 1.1
    retrieved_doc = [doc for doc, score in results_with_scores if score > similarity_threshold]

    # Manually retrieve documents first
    retrieved_docs = retriever.invoke(question)

    # if is_greeting_via_llm(question, selected_model):
    #     llm_greet = ChatOllama(model=selected_model, streaming=True, temperature=0.7)
    #     return llm_greet.invoke(question).content  #

    message_type = detect_message_type(question, selected_model)
    print(message_type)
    if message_type in ["greeting", "gratitude"]:
        llm = ChatOllama(model=selected_model, streaming=True, temperature=0.7)
        return llm.invoke(question).content
    else:
        if retrieved_doc:
            return "I am designed to assist with XYZ organization-related queries only. Please ask something related."

    if role != "employee":

        # Check the categories of matched docs
        public_found = any(doc.metadata.get('category') == "public" for doc in retrieved_docs)
        if public_found:
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            RAG_TEMPLATE = """
            Answer the question using ONLY the following context:
            - Be brief and to the point.
            - Limit the answer to 2–3 sentences max.
            - If the question is a greeting, respond with a friendly and short greeting.

            {context}

            Question: {question}
            """

            prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
            context_runnable = RunnableLambda(lambda _: context)

            chain = (
                {"context": context_runnable, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            return chain.invoke(question)
        else:
            return "You are not an authenticated person. Please connect with the HR team."
    else:
        RAG_TEMPLATE = """
        Answer the question using ONLY the following context:
        - Be brief and to the point.
        - Limit the answer to 2–3 sentences max.
        - If the question is a greeting, respond with a friendly and short greeting.

        {context}

        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

            # Invoke the chain
        answer = chain.invoke(question)

        return answer
    




