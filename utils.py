from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from pypdf import PdfReader
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq
import time

def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def create_docs(user_pdf_list, unique_id):
    docs = []
    for file in user_pdf_list:
        text = get_pdf_text(file)
        docs.append(Document(
            page_content=text,
            metadata={
                "name": file.name,
                "unique_id": unique_id
            },
        ))
    return docs

def create_embeddings_load_data():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def push_to_pinecone(api_key, environment, index_name, embeddings, docs):
    pc = Pinecone(api_key=api_key)
    indexes = [i['name'] for i in pc.list_indexes().get('indexes', [])]

    if index_name not in indexes:
        pc.create_index(
            name=index_name,
            dimension=384,  # for all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")  # adjust if needed
        )
        print("Waiting for index to be ready...")
        time.sleep(60)

    index = pc.Index(index_name)
    vectorstore = LangchainPinecone.from_documents(
        docs,
        embedding=embeddings,
        index_name=index_name,
        namespace="resumes"
    )

def pull_from_pinecone(api_key, environment, index_name, embeddings):
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    return LangchainPinecone.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
        namespace="resumes"
    )

def similar_docs(query, k, api_key, environment, index_name, embeddings, unique_id, use_filter=True):
    vectorstore = pull_from_pinecone(api_key, environment, index_name, embeddings)
    if use_filter:
        filter = {"unique_id": {"$eq": unique_id}}
    else:
        filter = {}
    return vectorstore.similarity_search_with_score(
        query=query,
        k=int(k),
        filter=filter
    )

def get_summary(current_doc):
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run([current_doc])
