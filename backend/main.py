import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import Modular Components
from backend.core.database import init_db
from backend.core.engine import HybridRetriever
from backend.controller.query_controller import router as api_router

# LlamaIndex Imports
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- CONFIG ---
load_dotenv()

# --- LIFESPAN (Startup Logic) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- SERVER STARTUP ---")
    
    # 1. Initialize Database
    init_db()

    # 2. Setup Models (Local Ollama)
    print("⚙️  Connecting to Local Ollama...")
    Settings.llm = Ollama(model="llama3", request_timeout=360.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # 3. Load Index from Storage
    if not os.path.exists("storage"):
        print("🔴 [ERROR] Storage not found. Run ingest.py first!")
        yield
        return

    print("⏳ Loading Index from Disk...")
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context)
    
    # 4. Initialize Custom Hybrid Retriever
    all_nodes = list(index.docstore.docs.values())
    vector_store = index.vector_store
    
    retriever = HybridRetriever(
        vector_store=vector_store,
        nodes=all_nodes,
        top_k=5,
        rerank_top_n=20
    )
    
    # 5. Attach Engine to Global App State
    app.state.query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ]
    )
    
    print("🚀 [READY] RAG Engine Online (Hybrid + Parent-Child + Local Ollama)")
    yield
    # Cleanup
    app.state.query_engine = None

# --- APP SETUP ---
app = FastAPI(title="AuditAI RAG API", lifespan=lifespan)

# CORS (Allow Frontend Access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routes
app.include_router(api_router)