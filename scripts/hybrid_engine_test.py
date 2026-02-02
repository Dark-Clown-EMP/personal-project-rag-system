import sys
import os

# --- PATH HACK (To find 'core') ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.engine import HybridRetriever
import chromadb
from llama_index.core import Settings, get_response_synthesizer, PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import TextNode
import nest_asyncio

# Fix for nested event loops
nest_asyncio.apply()

# --- THE RESUME PROMISE: "Citation-First Generation" ---
# This prompt forces the "Auditor" persona.
citation_prompt_str = (
    "You are a strict financial auditor reviewing technical documentation.\n"
    "Your goal is to answer the user's query using ONLY the provided context.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "RULES:\n"
    "1. You must specific source clauses for every fact (e.g., [Section 4.1], [Para 12]).\n"
    "2. If the context does not contain the answer, explicitly state: 'I cannot find this information in the provided text.'\n"
    "3. Do not use outside knowledge. Do not hallucinate.\n"
    "4. Keep the answer professional, concise, and audit-ready.\n\n"
    "Query: {query_str}\n"
    "Answer: "
)

def run_test():
    print("--- 1. LOADING RESOURCES (Using GPU if available) ---")
    # Llama 3 is required here.
    llm = Ollama(model="llama3", request_timeout=360.0)
    
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    print("--- 2. CONNECTING TO DB ---")
    db = chromadb.PersistentClient(path="./chroma_local_db")
    chroma_collection = db.get_collection("basel_risk_local")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    print("--- 3. BUILDING HYBRID RETRIEVER ---")
    # Fetch data for BM25
    result = chroma_collection.get()
    nodes = [TextNode(text=t, id_=i) for t, i in zip(result['documents'], result['ids'])]
    
    retriever = HybridRetriever(vector_store=vector_store, nodes=nodes, top_k=5)
    
    # --- 4. INJECTING THE PROMPT ---
    qa_template = PromptTemplate(citation_prompt_str)
    
    synthesizer = get_response_synthesizer(
        response_mode="compact",
        text_qa_template=qa_template # <--- The Resume Logic
    )
    
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer
    )
    
    # --- 5. EXECUTION ---
    query = "What is the minimum Common Equity Tier 1 capital ratio? Cite the specific paragraph."
    
    print(f"\nQUERY: {query}")
    print("Thinking...")
    
    response = query_engine.query(query)
    
    print("\n--- LLM ANSWER ---")
    print(response)
    
    print("\n--- CITATION VERIFICATION ---")
    # We print the source nodes to verify the LLM isn't hallucinating the paragraph numbers
    for i, node in enumerate(response.source_nodes):
        print(f"[{i+1}] Score: {node.score:.4f} | Snippet: {node.node.get_content()[:50]}...")

if __name__ == "__main__":
    run_test()