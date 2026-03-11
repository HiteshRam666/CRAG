import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.retriever import vector_store
import os 
from dotenv import load_dotenv
load_dotenv()

ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(file_bytes: bytes):
    docs = fitz.open(stream = file_bytes, filetype='pdf') 
    text = "" 
    for page in docs:
        text += page.get_text() 
    return text 

# def ingest_pdf(file_bytes: bytes):
#     raw_text = extract_text_from_pdf(file_bytes) 

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size = 800, 
#         chunk_overlap = 200 
#     )

#     docs = [Document(page_content = raw_text)] 
#     chunks = splitter.split_documents(docs) 

#     vector_store.add_documents(chunks)

def ingest_pdf(file_bytes: bytes, thread_id: str):
    raw_text = extract_text_from_pdf(file_bytes)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800, 
        chunk_overlap = 200
    )

    chunks = splitter.split_text(raw_text) 

    vector_store.add_texts(
        texts = chunks, 
        metadatas=[{"thread_id": thread_id} for _ in chunks]
    )