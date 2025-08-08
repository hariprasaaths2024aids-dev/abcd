import tempfile
import requests
import fitz  # PyMuPDF
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.llms import HuggingFaceEndpoint

def process_documents(doc_url: str, questions: list) -> list:
    response = requests.get(doc_url)
    if response.status_code != 200:
        raise Exception("Failed to download document")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name

    text = ""
    with fitz.open(tmp_path) as doc:
        for page in doc:
            text += page.get_text()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

    vectorstore = FAISS.from_documents(documents, embeddings)
    
    retriever = vectorstore.as_retriever()

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-70B-Instruct",
        temperature=0.7,
        model_kwargs={"max_new_tokens": 500}
    )

    results = []
    for question in questions:
        docs = retriever.get_relevant_documents(question)
        context = " ".join([doc.page_content for doc in docs[:3]])
        prompt = f"""Answer the question based on the context:
Context: {context}
Question: {question}
Answer:"""
        answer = llm.invoke(prompt)
        results.append(answer.strip())
    return results
