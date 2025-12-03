import os

from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

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


def create_embeddings():
    """Create embedding model using Google Generative AI."""
    print(f"{GREEN}[INFO] Initializing Google Generative AI embeddings...{RESET}")
    gemini_embeddings =  GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    return gemini_embeddings

def run_direct_llm_prompt():
    print(f"{GREEN}[INFO] Running direct LLM prompt only (no RAG)...{RESET}")

    gemini_llm = GoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )

    query = "What is Pinecone in machine learning?"
    prompt = PromptTemplate.from_template(template=query)
    chain = prompt | gemini_llm
    result = chain.invoke(input={})
    print(result)

def main():
    gemini_embeddings = create_embeddings()

    gemini_llm = GoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )
    print(f"{GREEN}[INFO] Connecting to Pinecone vector store...{RESET}")
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME, embedding=gemini_embeddings
    )

    print(f"{GREEN}[INFO] Loading retrieval QA prompt from LangChain Hub...{RESET}")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    print(f"{GREEN}[INFO] Creating document combination chain...{RESET}")
    combined_docs_chain = create_stuff_documents_chain(
        gemini_llm,
        retrieval_qa_chat_prompt
    )

    print(f"{GREEN}[INFO] Creating retrieval chain...{RESET}")

    # Change: configure retriever to fetch top 8 chunks, not only top 4 (default).
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combined_docs_chain
    )

    chat_history = []
    print(f"{GREEN}[INFO] (type 'quit' to exit){RESET}")
    while True:
        user_query = input(f"{GREEN}Your question: {RESET}")

        if user_query.lower() in ['quit', 'exit', 'q']:
            print(f"{GREEN}Goodbye!{RESET}")
            break

        if not user_query.strip():
            continue

        try:
            result = retrieval_chain.invoke({
                "input": user_query,
                "chat_history": chat_history
            })
            print(f"\n{GREEN}[ANSWER]:{RESET}")
            print(f"{result['answer']}\n")
            print(f"{GREEN}[Sources]: {len(result['context'])} documents used{RESET}\n")
            chat_history.append(HumanMessage(content=user_query))
            chat_history.append(AIMessage(content=result['answer']))
        except Exception as e:
            print(f"{RED}[ERROR]: {e}{RESET}\n")


def format_docs(docs):
    """Format documents by joining them with newlines."""
    return "\n\n".join(doc.page_content for doc in docs)

def run_custom_rag_with_lcel():
    from langchain_core.runnables import RunnablePassthrough
    """Run RAG using custom prompt and LCEL (LangChain Expression Language)."""
    print(f"{GREEN}[INFO] Running custom RAG with LCEL...{RESET}")

    gemini_embeddings = create_embeddings()
    gemini_llm = GoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )

    print(f"{GREEN}[INFO] Connecting to Pinecone vector store...{RESET}")
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME, embedding=gemini_embeddings
    )

    # Create custom prompt template
    template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say you don't know. Don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say thanks for asking at the end of the answer.


{context}

Question: {question}

Helpful answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template)

    # Get retriever from vector store
    retriever = vectorstore.as_retriever()

    # Build the RAG chain using LCEL
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": RunnablePassthrough()
        }
        | custom_rag_prompt
        | gemini_llm
    )

    # Test query
    query = "What is Pinecone in machine learning?"
    print(f"{GREEN}[INFO] Running query: {query}{RESET}")

    result = rag_chain.invoke(query)

    print(f"\n{GREEN}[ANSWER]:{RESET}")
    print(f"{result}\n")

    return result


if __name__ == "__main__" :
    # run_direct_llm_prompt()
    # main()
    run_custom_rag_with_lcel()
