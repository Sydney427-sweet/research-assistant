import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DOCS_DIR = "docs"
CHROMA_DIR = ".chroma"

def load_documents():
    docs = []
    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".pdf"):
            print(f"Loading: {filename}")
            loader = PyPDFLoader(os.path.join(DOCS_DIR, filename))
            docs.extend(loader.load())
    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks from {len(docs)} pages")
    return chunks

def store_chunks(chunks):
    print("Embedding and storing chunks locally...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"  # fast, lightweight, great quality
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB")
    return vectorstore

if __name__ == "__main__":
    docs = load_documents()
    if not docs:
        print("❌ No PDFs found in /docs folder")
    else:
        chunks = chunk_documents(docs)
        store_chunks(chunks)
