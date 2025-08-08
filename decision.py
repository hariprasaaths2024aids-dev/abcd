import os
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

def evaluate_with_llm(query: str, vectorstore):
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-70b-8192"
    )

    template = '''You are a helpful assistant for health insurance-related document analysis.
Use only the provided context to answer the question. Be concise, accurate, and include a justification.

Context: {context}
Question: {question}

Answer with a JSON object: {{"justification": "your justification here"}}'''

    prompt = ChatPromptTemplate.from_template(template)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

    return chain.run(query)