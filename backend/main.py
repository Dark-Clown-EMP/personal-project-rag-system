import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- PATH HACK (To find 'core') ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import HybridRetriever
from llama_index.core import Settings, get_response_synthesizer, PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import TextNode
import chromadb

# --- GLOBAL STATE ---
app_state = {"query_engine": None}

# --- LIFESPAN MANAGER (Startup Logic) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- SERVER STARTUP: Loading AI Models... ---")
    
    # 1. Load Models
    llm = Ollama(model="llama3", request_timeout=360.0)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # 2. Connect to Database
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_local_db")
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_collection("basel_risk_local")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # 3. Load Data for BM25
    print("Loading nodes for BM25 Index...")
    result = chroma_collection.get()
    nodes = [TextNode(text=t, id_=i) for t, i in zip(result['documents'], result['ids'])]
    
    # 4. Initialize Engine
    # Note: We are using the 'rerank_top_n' param for the Cross-Encoder
    retriever = HybridRetriever(
        vector_store=vector_store, 
        nodes=nodes, 
        top_k=5, 
        rerank_top_n=20
    )
    
    # 5. Auditor Prompt
    citation_prompt = (
        "You are a strict financial auditor.\n"
        "Answer the query based ONLY on the provided context.\n"
        "RULES:\n"
        "1. Cite specific sections [e.g., Para 4.1] for every claim.\n"
        "2. If the answer is not in the text, say 'I cannot find this information'.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    
    synthesizer = get_response_synthesizer(
        response_mode="compact",
        text_qa_template=PromptTemplate(citation_prompt)
    )
    
    app_state["query_engine"] = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer
    )
    
    print("--- SERVER READY: AI Engine Loaded ---")
    yield
    app_state.clear()

# --- API SETUP ---
app = FastAPI(title="Basel III RAG API", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/query", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    if not app_state["query_engine"]:
        raise HTTPException(status_code=503, detail="Engine is still loading...")
    
    try:
        response = app_state["query_engine"].query(request.query)
        
        # Format sources
        source_list = []
        for node in response.source_nodes:
            # Clean text and include score
            text = node.node.get_content().replace("\n", " ")[:200]
            # Handle score safely (Cross-Encoder scores can be float or numpy)
            score = float(node.score) if node.score is not None else 0.0
            source_list.append(f"[Score: {score:.2f}] {text}...")
            
        return QueryResponse(
            answer=str(response),
            sources=source_list
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))