import os
import nest_asyncio
from dotenv import load_dotenv

# --- CORRECT IMPORTS ---
# 1. The LLM (Brain) is now 'GoogleGenAI'
from llama_index.llms.google_genai import GoogleGenAI
# 2. The Embedder is now 'GoogleGenAIEmbedding'
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

nest_asyncio.apply()
load_dotenv()

# Verify Keys
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in .env")

print("--- 1. INITIALIZING GEMINI 1.5 PRO ---")

# FIX: Use 'GoogleGenAI' class
llm = GoogleGenAI(
    model="models/gemini-1.5-pro",
    api_key=os.getenv("GOOGLE_API_KEY")
) 

# FIX: Use 'GoogleGenAIEmbedding' class
embed_model = GoogleGenAIEmbedding(
    model_name="models/text-embedding-004",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Bind them
Settings.llm = llm
Settings.embed_model = embed_model

print("--- 2. PARSING WITH LLAMAPARSE ---")
parser = LlamaParse(
    result_type="markdown",
    verbose=True,
    language="en",
)

# Update this path if needed
pdf_path = "./data/bcbs189.pdf"
documents = parser.load_data(pdf_path)
print(f"Parsed {len(documents)} pages.")

print("--- 3. STORING IN CHROMA (Persistent) ---")
# Saving to 'chroma_pro_db'
db = chromadb.PersistentClient(path="./chroma_pro_db")
chroma_collection = db.get_or_create_collection("basel_risk_pro")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("--- 4. INDEXING ---")
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

print("--- 5. TEST QUERY ---")
query_engine = index.as_query_engine()
response = query_engine.query("What is the minimum Common Equity Tier 1 capital ratio? Provide the specific percentage.")

print("\nRESPONSE FROM GEMINI 1.5 PRO:")
print(response)