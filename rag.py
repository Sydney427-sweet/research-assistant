from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_DIR = ".chroma"

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # fetch top 4 most relevant chunks
    )

    prompt = PromptTemplate.from_template("""
You are a helpful research assistant. Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have enough information to answer that."
Always cite which part of the context you used.

Context:
{context}

Question:
{question}

Answer:
""")

    llm = ChatOllama(model="llama3.2", temperature=0)

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[Chunk {i+1}]: {doc.page_content}"
            for i, doc in enumerate(docs)
        )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

def ask(question: str):
    vectorstore = load_vectorstore()
    chain, retriever = build_rag_chain(vectorstore)

    print(f"\n🔍 Question: {question}")
    print("⏳ Retrieving and generating...\n")

    # get retrieved chunks for transparency
    chunks = retriever.invoke(question)
    print("📄 Retrieved chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  [{i+1}] ...{chunk.page_content[:120]}...")

    # get the answer
    answer = chain.invoke(question)
    print(f"\n💬 Answer:\n{answer}")
    return answer, chunks

if __name__ == "__main__":
    question = input("Ask a question about your documents: ")
    ask(question)
