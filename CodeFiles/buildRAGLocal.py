import os
import nest_asyncio
from dotenv import load_dotenv

# --- IMPORTS FOR LOCAL STACK ---
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

nest_asyncio.apply()
load_dotenv()

# Verify LlamaCloud Key (Still needed for high-quality PDF parsing)
if not os.getenv("LLAMA_CLOUD_API_KEY"):
    raise ValueError("LLAMA_CLOUD_API_KEY not found in .env")

print("--- 1. INITIALIZING LOCAL STACK ---")

# A. The Brain (Running on your PC via Ollama)
# request_timeout is set high because local PCs can be slower than cloud API
print("Connecting to Local Ollama (llama3)...")
llm = Ollama(model="llama3", request_timeout=360.0)

# B. The Embedder (Running on your PC via HuggingFace)
# This downloads a specific high-performance model locally
print("Loading Local Embedding Model (bge-small-en)...")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Bind them to Settings
Settings.llm = llm
Settings.embed_model = embed_model

print("--- 2. PARSING WITH LLAMAPARSE (Cloud) ---")
parser = LlamaParse(
    result_type="markdown",
    verbose=True,
    language="en",
)

# Update this path to where your PDF actually is
pdf_path = "./data/bcbs189.pdf" 
documents = parser.load_data(pdf_path)
print(f"Parsed {len(documents)} pages.")

print("--- 3. STORING IN CHROMA (Persistent) ---")
# Saving to a NEW local database folder to avoid conflicts
db = chromadb.PersistentClient(path="./chroma_local_db")
chroma_collection = db.get_or_create_collection("basel_risk_local")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("--- 4. INDEXING (Local Embeddings) ---")
# This might take a minute on your PC
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

print("--- 5. TEST QUERY (Local Inference) ---")
query_engine = index.as_query_engine()
print("Asking Llama 3...")
response = query_engine.query("What is the minimum Common Equity Tier 1 capital ratio? Provide the specific percentage.")

print("\nRESPONSE FROM LOCAL LLAMA 3:")
print(response)