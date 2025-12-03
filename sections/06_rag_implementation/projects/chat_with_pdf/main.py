import os
from typing import List

from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not set in environment or .env file")

FAISS_STORE_PATH = "vector_store/faiss_index"
PDF_FILE_PATH = "data/ReAct2.pdf"

def load_pdf_text(pdf_path: str) -> List[Document]:
    print(f"{GREEN}[INFO] Loading PDF from: {pdf_path}{RESET}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"{GREEN}[INFO] Loaded {len(documents)} pages.{RESET}")
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    print(f"{GREEN}[INFO] Splitting documents into chunks...{RESET}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"{GREEN}[INFO] Created {len(chunks)} chunk(s).{RESET}")
    return chunks

def get_embeddings():
    print(f"{GREEN}[INFO] Creating HuggingFace embeddings object...{RESET}")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

def build_or_load_vector_store(chunks: List[Document], embeddings, store_path: str) -> FAISS:
    if os.path.exists(store_path):
        print(f"{GREEN}[INFO] Loading existing FAISS vector store from {store_path}{RESET}")
        vectorstore = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"{GREEN}[INFO] Building new FAISS vector store...{RESET}")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        vectorstore.save_local(store_path)
        print(f"{GREEN}[INFO] Saved FAISS vector store to {store_path}{RESET}")
    return vectorstore

def rephrase_query(llm, user_query: str, chat_history: List) -> str:
    """Rephrase the user query to make it more effective for retrieval."""
    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at rephrasing questions to make them more effective for document retrieval.
Given a user question (and optionally chat history for context), rephrase it into a clear, standalone search query that would retrieve the most relevant documents.

Rules:
- Make the query clear and specific
- Include important keywords
- Remove conversational elements
- If the question refers to previous context, incorporate that context into a standalone query
- Keep it concise but comprehensive"""),
        ("human", "Chat History: {chat_history}\n\nUser Question: {question}\n\nRephrased Query:")
    ])

    chain = rephrase_prompt | llm
    rephrased = chain.invoke({
        "question": user_query,
        "chat_history": str(chat_history) if chat_history else "None"
    })

    return rephrased.strip()


def main():
    # 1. Load PDF → Extract Text
    documents = load_pdf_text(PDF_FILE_PATH)

    # 2. Split Text into Chunks
    chunks = split_documents(documents)

    # 3. Embed Chunks and 4. Build/Load the Vector Store
    embeddings = get_embeddings()
    vectorstore = build_or_load_vector_store(chunks, embeddings, FAISS_STORE_PATH)

    # 5. Create LLM
    print(f"{GREEN}[INFO] Initializing Gemini LLM...{RESET}")
    llm = GoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )

    # 6. Load retrieval QA prompt from LangChain Hub
    print(f"{GREEN}[INFO] Loading retrieval QA prompt from LangChain Hub...{RESET}")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # 7. Create document combination chain
    print(f"{GREEN}[INFO] Creating document combination chain...{RESET}")
    combined_docs_chain = create_stuff_documents_chain(
        llm,
        retrieval_qa_chat_prompt
    )

    # 8. Create retrieval chain with retriever configured to fetch top 10 chunks
    print(f"{GREEN}[INFO] Creating retrieval chain...{RESET}")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combined_docs_chain
    )

    # 9. Chat loop with history
    chat_history = []
    print(f"{GREEN}[INFO] (Type 'quit' to exit){RESET}")
    while True:
        user_query = input(f"{GREEN}Your question: {RESET}")

        if user_query.lower() in ['quit', 'exit', 'q']:
            print(f"{GREEN}Goodbye!{RESET}")
            break

        if not user_query.strip():
            continue

        try:
            # Rephrase the query for better retrieval
            print(f"{GREEN}[INFO] Rephrasing query...{RESET}")
            rephrased_query = rephrase_query(llm, user_query, chat_history)
            print(f"{GREEN}[REPHRASED QUERY]: {rephrased_query}{RESET}\n")

            # Use the rephrased query for retrieval
            result = retrieval_chain.invoke({
                "input": rephrased_query,
                "chat_history": chat_history
            })
            print(f"\n{GREEN}[ANSWER]:{RESET}")
            print(f"{result['answer']}\n")
            print(f"{GREEN}[Sources]: {len(result['context'])} documents used{RESET}\n")

            # Update chat history with original query
            chat_history.append(HumanMessage(content=user_query))
            chat_history.append(AIMessage(content=result['answer']))
        except Exception as e:
            print(f"{RED}[ERROR]: {e}{RESET}\n")

if __name__ == "__main__":
    main()
