from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from reranker import rerank

load_dotenv()

CHROMA_DIR = ".chroma"

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}  # fetch 10 so reranker has room to work
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

    def retrieve_and_rerank(question):
        chunks = retriever.invoke(question)
        reranked = rerank(question, chunks, top_k=4)
        return "\n\n---\n\n".join(
            f"[Chunk {i+1}]: {chunk.page_content}"
            for i, chunk in enumerate(reranked)
        )

    chain = (
        {"context": retrieve_and_rerank, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

def ask(question: str):
    vectorstore = load_vectorstore()
    chain, retriever = build_rag_chain(vectorstore)

    print(f"\n🔍 Question: {question}")
    print("⏳ Retrieving, reranking, and generating...\n")

    chunks = retriever.invoke(question)
    reranked_chunks = rerank(question, chunks, top_k=4)
    answer = chain.invoke(question)

    print(f"\n💬 Answer:\n{answer}")
    return answer, reranked_chunks

if __name__ == "__main__":
    question = input("Ask a question about your documents: ")
    ask(question)
