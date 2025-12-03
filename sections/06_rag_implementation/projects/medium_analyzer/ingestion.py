import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not set in environment or .env file")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise EnvironmentError("PINECONE_API_KEY not set in environment or .env file")

INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "medium-blogs-embeddings")

def load_documents(file_path: str):
    """Load documents from a text file."""
    print(f"{GREEN}[INFO] Loading document from: {file_path}{RESET}")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    print(f"{GREEN}[INFO] Loaded {len(documents)} document(s){RESET}")

    return documents

def split_documents(documents: List[Document]):
    """Split documents into chunks."""
    print(f"{GREEN}[INFO] Starting the document splitting process...{RESET}")

    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    print(f"{GREEN}[INFO] Successfully split into {len(texts)} chunk(s).{RESET}")
    for idx, doc in enumerate(texts):
        doc_len = len(doc.page_content) if hasattr(doc, "page_content") else "N/A"
        print(f"{GREEN}[INFO] Chunk {idx}: length = {doc_len}{RESET}")
    return texts

def create_embeddings():
    """Create embedding model using Google Generative AI."""
    print(f"{GREEN}[INFO] Initializing Google Generative AI embeddings...{RESET}")
    gemini_embeddings =  GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    return gemini_embeddings

def ingest_to_pinecone(texts: List[Document], embeddings):
    """Ingest document chunks into Pinecone vector store."""
    print(f"{GREEN}[INFO] Ingesting {len(texts)} chunks into Pinecone...{RESET}")

    vector_store = PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

    print(f"{GREEN}[INFO] ✓ Finished ingesting documents into Pinecone!{RESET}")

    return vector_store

def main():
    file_path = str(Path("data/MediumBlog1.txt"))
    documents = load_documents(file_path)
    texts = split_documents(documents)
    gemini_embeddings = create_embeddings()
    vector_store = ingest_to_pinecone(texts, gemini_embeddings)


if __name__ == '__main__':
    print("Ingesting using Gemini embeddings...")
    main()
