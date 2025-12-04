
import asyncio
import datetime as dt
import json
import os
import ssl
from typing import Any, Dict, List

import certifi
import tiktoken
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

from logger import Colors, log_error, log_header, log_info, log_success, log_warning

# ============================================================================
# ENVIRONMENT AND API KEY SETUP
# ============================================================================

load_dotenv()

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    log_error("GOOGLE_API_KEY not set in environment or .env file")
    raise EnvironmentError("GOOGLE_API_KEY not set in environment or .env file")

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    log_error("TAVILY_API_KEY not set in environment or .env file")
    raise EnvironmentError("TAVILY_API_KEY not set in environment or .env file")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    log_error("PINECONE_API_KEY not set in environment or .env file")
    raise EnvironmentError("PINECONE_API_KEY not set in environment or .env file")

PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "documentation")
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "documentation-docs")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_embeddings():
    """Initialize HuggingFace embeddings."""
    log_info("Creating HuggingFace embeddings object...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
    )
    log_success("Embeddings object created successfully!")
    return embeddings

# --------- tiktoken Length Function --------- #
enc = tiktoken.get_encoding("cl100k_base")
def length_function(text: str) -> int:
    """Calculate token length using tiktoken."""
    return len(enc.encode(text))

# --------- Text Splitter --------- #
def get_text_splitter():
    """Initialize text splitter with custom chunk settings and tiktoken length."""
    log_info("Creating text splitter...")
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=3000,          # Break docs into 3000-char chunks
        chunk_overlap=300,        # 300-char overlap to preserve context
        length_function=length_function,
    )
    log_success("Text splitter created successfully!")
    return splitter

# --------- LLM --------- #
def get_llm():
    """Initialize Google Generative AI LLM."""
    log_info("Creating ChatGoogleGenerativeAI model...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        api_key=GOOGLE_API_KEY,
    )
    log_success("LLM initialized successfully!")
    return llm

# ============================================================================
# PINECONE SETUP AND CONNECTION
# ============================================================================

def get_pinecone_vectorstore() -> PineconeVectorStore:
    """
    Connect to Pinecone vector store.
    Creates connection to existing index.

    Returns:
        PineconeVectorStore instance
    """
    log_info(f"Connecting to Pinecone index: {PINECONE_INDEX_NAME}")

    embeddings = get_embeddings()

    # Connect to existing Pinecone index
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=PINECONE_NAMESPACE,
    )

    log_success(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")
    return vectorstore

def verify_pinecone_index():
    """
    Verify Pinecone index exists before processing.

    Returns:
        True if index exists and is ready, False otherwise
    """
    try:
        log_info(f"Verifying Pinecone index: {PINECONE_INDEX_NAME}...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_list = pc.list_indexes()

        index_exists = any(idx.name == PINECONE_INDEX_NAME for idx in index_list)

        if index_exists:
            log_success(f"✓ Pinecone index '{PINECONE_INDEX_NAME}' is ready")
            return True
        else:
            log_error(f"✗ Pinecone index '{PINECONE_INDEX_NAME}' not found")
            log_info(f"Available indexes: {[idx.name for idx in index_list]}")
            return False
    except Exception as e:
        log_error(f"Failed to verify Pinecone index: {str(e)}")
        return False

# ============================================================================
# TAVILY TOOLS / WEB SEARCH
# ============================================================================

def get_tavily_tools():
    """Initialize Tavily web search tools."""
    log_info("Initializing Tavily tools...")

    tavily_extract = TavilyExtract(api_key=TAVILY_API_KEY)
    tavily_map = TavilyMap(
        api_key=TAVILY_API_KEY,
        max_depth=5,        # Discover URLs 2 levels deep
        max_breadth=20,     # Max 20 URLs per level
        max_pages=1000       # Total URL cap
    )
    tavily_crawl = TavilyCrawl(api_key=TAVILY_API_KEY)

    log_success("Tavily tools initialized successfully!")
    return {
        "extract": tavily_extract,
        "map": tavily_map,
        "crawl": tavily_crawl,
    }

# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

def process_map_results(results: Dict) -> List[str]:
    """
    Extract URLs from TavilyMap results.
    TavilyMap returns discovered URLs, not raw content.

    Args:
        results: Map results from Tavily

    Returns:
        List of discovered URLs
    """
    log_info(f"Processing TavilyMap results...")

    urls = []
    for result in results.get("results", []):
        if "url" in result:
            urls.append(result["url"])
            log_info(f"  📍 Discovered: {result['url']}")

    log_success(f"Found {len(urls)} URLs from site map")
    return urls

async def extract_content_batch(
    tavily_extract: TavilyExtract,
    urls: List[str],
    batch_size: int = 5
) -> List[Document]:
    """
    Extract content from URLs using TavilyExtract (targeted extractor).
    Processes URLs in batches to respect API rate limits.

    Args:
        tavily_extract: TavilyExtract tool instance
        urls: List of URLs to extract from
        batch_size: URLs per concurrent batch

    Returns:
        List of Document objects
    """
    log_header("EXTRACTING CONTENT FROM DISCOVERED URLS")
    log_info(f"Extracting content from {len(urls)} URLs in batches of {batch_size}")

    # Split URLs into batches for concurrent processing
    batches = [urls[i:i + batch_size] for i in range(0, len(urls), batch_size)]
    log_info(f"📦 Split into {len(batches)} extraction batches")

    all_documents = []

    async def extract_batch(batch: List[str], batch_num: int):
        """Extract content from a batch of URLs."""
        try:
            batch_docs = []
            for url in batch:
                try:
                    result = tavily_extract.invoke({"url": url})
                    content = result.get("raw_content", "")

                    if content.strip():
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": url,
                                "title": result.get("title", ""),
                                "extracted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                            }
                        )
                        batch_docs.append(doc)
                        log_info(f"  ✓ Extracted from {url}")
                    else:
                        log_warning(f"  ✗ Empty content from {url}")

                except Exception as e:
                    log_error(f"  ✗ Failed to extract {url}: {str(e)}")

            log_success(f"Batch {batch_num}/{len(batches)} complete ({len(batch_docs)} documents extracted)")
            return batch_docs

        except Exception as e:
            log_error(f"Batch {batch_num} failed: {str(e)}")
            return []

    # Process batches concurrently
    tasks = [extract_batch(batch, i + 1) for i, batch in enumerate(batches)]
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Flatten results from all batches
    for result in batch_results:
        if isinstance(result, list):
            all_documents.extend(result)

    log_success(f"Extracted {len(all_documents)} documents total")
    return all_documents

def process_crawl_results(results: Dict) -> List[Document]:
    """
    Convert Tavily crawl results to LangChain Documents.
    Use when crawling directly instead of map+extract.

    Args:
        results: Crawl results from Tavily

    Returns:
        List of Document objects
    """
    log_info(f"Processing {len(results.get('results', []))} crawl results...")

    documents = []
    for doc in results.get("results", []):
        try:
            content = doc.get("raw_content", "")
            if not content.strip():
                log_warning(f"Empty content from {doc.get('url')}")
                continue

            document = Document(
                page_content=content,
                metadata={
                    "source": doc.get("url", "unknown"),
                    "title": doc.get("title", ""),
                    "crawled_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                }
            )
            documents.append(document)
        except Exception as e:
            log_warning(f"Failed to process document: {str(e)}")

    log_success(f"Created {len(documents)} documents from crawl results")
    return documents

def filter_documents(documents: List[Document], min_length: int = 50) -> List[Document]:
    """
    Filter out low-quality documents.

    Args:
        documents: List of documents to filter
        min_length: Minimum content length in characters

    Returns:
        Filtered documents
    """
    log_info(f"Filtering {len(documents)} documents (min_length={min_length})...")

    filtered = [
        doc for doc in documents
        if len(doc.page_content.strip()) >= min_length
    ]

    removed = len(documents) - len(filtered)
    log_success(f"Filtered: {len(filtered)} kept, {removed} removed")
    return filtered

# ============================================================================
# CHUNKING AND PINECONE INDEXING
# ============================================================================

async def chunk_and_index_pinecone_async(
    documents: List[Document],
    vectorstore: PineconeVectorStore,
    batch_size: int = 500
):
    """
    Chunk documents and add them to Pinecone asynchronously in batches.
    Uses aadd_documents for concurrent indexing to Pinecone.

    Args:
        documents: List of Document objects to chunk and index
        vectorstore: PineconeVectorStore instance
        batch_size: Chunks per concurrent batch (default 500)
    """
    log_header("DOCUMENT CHUNKING PHASE")
    # FIX: log_info() by removing color argument for compatibility
    log_info(f"✂️  Chunking {len(documents)} documents with 3000 chunk size and 300 overlap")

    # Stage 1: Split documents into chunks
    text_splitter = get_text_splitter()
    chunked_docs = text_splitter.split_documents(documents)
    log_success(
        f"Text Splitter: Created {len(chunked_docs)} chunks from {len(documents)} documents"
    )

    # Stage 2: Create batches for concurrent indexing
    log_header("PINECONE INDEXING PHASE")
    # FIX: log_info() by removing color argument for compatibility
    log_info(f"📚 Pinecone Indexing: Preparing to add {len(chunked_docs)} chunks to Pinecone")

    batches = [
        chunked_docs[i : i + batch_size]
        for i in range(0, len(chunked_docs), batch_size)
    ]

    log_info(
        f"📦 Pinecone Indexing: Split into {len(batches)} batches of {batch_size} chunks each"
    )

    # Stage 3: Add batches concurrently using aadd_documents to Pinecone
    async def add_batch_to_pinecone(batch: List[Document], batch_num: int):
        """Add a batch of chunks to Pinecone asynchronously."""
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"Pinecone Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} chunks)"
            )
            return True
        except Exception as e:
            log_error(f"Pinecone Indexing: Failed to add batch {batch_num} - {e}")
            return False

    # Process all batches concurrently
    tasks = [add_batch_to_pinecone(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"Pinecone Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"Pinecone Indexing: Processed {successful}/{len(batches)} batches successfully"
        )

# ============================================================================
# MAIN WORKFLOW - COMPLETE PIPELINE
# ============================================================================

async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    # Stage 0: Verify Pinecone is ready
    log_header("Stage 0: Verifying Pinecone Connection")
    if not verify_pinecone_index():
        log_error("Pinecone index not available. Exiting.")
        return

    # Initialize Pinecone vector store
    vectorstore = get_pinecone_vectorstore()

    # ========== OPTION 1: MAP + EXTRACT (Recommended for structured sites) ==========
    log_header("Stage 1: Discovering URLs with TavilyMap")
    tavily_tools = get_tavily_tools()
    tavily_map = tavily_tools["map"]
    tavily_extract = tavily_tools["extract"]

    # map_results = tavily_map.invoke({
    #     "url": "https://python.langchain.com/",
    # })
    # discovered_urls = process_map_results(map_results)
    # log_info(f"🗺️  TavilyMap: Discovered {len(discovered_urls)} URLs")

    # # Stage 2: Extract content from discovered URLs
    # log_header("Stage 2: Extracting Content with TavilyExtract")
    # extracted_docs = await extract_content_batch(tavily_extract, discovered_urls, batch_size=5)
    # log_info(f"📄 TavilyExtract: Extracted {len(extracted_docs)} documents")

    # ========== OPTION 2: CRAWL (Alternative - deep recursive crawling) ==========
    # Uncomment below to use TavilyCrawl instead of map+extract
    log_header("Stage 1: Crawling Documentation with TavilyCrawl")
    tavily_crawl = tavily_tools["crawl"]
    crawl_results = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",
        "max_depth": 5,
        "extract_depth": "advanced",
    })
    extracted_docs = process_crawl_results(crawl_results)
    log_info(f"🕷️  TavilyCrawl: Crawled {len(extracted_docs)} documents")

    # Stage 3: Filter documents
    log_header("Stage 3: Filtering Documents")
    filtered_docs = filter_documents(extracted_docs, min_length=50)
    log_info(f"🔍 Filter: Kept {len(filtered_docs)} quality documents")

    # Stage 4: Chunk and index asynchronously with batching to Pinecone
    log_header("Stage 4: Chunking and Async Indexing to Pinecone")
    await chunk_and_index_pinecone_async(filtered_docs, vectorstore, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    # FIX: log_info() by removing Colors.BOLD argument for compatibility
    log_info("📊 Summary:")
    # log_info(f"   • URLs discovered: {len(discovered_urls)}")
    log_info(f"   • Documents extracted: {len(extracted_docs)}")
    log_info(f"   • Documents after filtering: {len(filtered_docs)}")
    log_info(f"   • Pinecone index: {PINECONE_INDEX_NAME}")
    log_info(f"   • Pinecone namespace: {PINECONE_NAMESPACE}")

    await asyncio.sleep(1)
    log_success("Done!")

if __name__ == "__main__":
    asyncio.run(main())
