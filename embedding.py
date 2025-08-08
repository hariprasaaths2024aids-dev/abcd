import os
import fitz  # PyMuPDF
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def load_document(pdf_path: str):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def create_vectorstore(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")
    db = FAISS.from_documents(documents, embeddings)
    return db