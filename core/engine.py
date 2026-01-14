import chromadb
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import VectorStoreQuery
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder # <--- NEW IMPORT
import nltk

# --- NLTK Setup ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_store, nodes, top_k=5, rerank_top_n=20):
        """
        top_k: The final number of chunks to give the LLM (e.g., 5).
        rerank_top_n: How many candidates to send to the Cross-Encoder (e.g., 20).
        """
        super().__init__()
        self.vector_store = vector_store
        self.nodes = nodes
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n # <--- NEW PARAMETER
        
        print("Building BM25 Index from stored text...")
        tokenized_corpus = [word_tokenize(node.get_content()) for node in nodes]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 Index Ready.")

        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # --- LOAD CROSS-ENCODER (Resume Point #4) ---
        print("Loading Cross-Encoder Model...")
        # 'ms-marco-MiniLM-L-6-v2' is the industry standard for fast re-ranking
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("Cross-Encoder Ready.")

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query_str = query_bundle.query_str
        
        # --- STAGE 1: HYBRID RETRIEVAL (RRF) ---
        # We fetch MORE results than we need (rerank_top_n) to give the re-ranker options
        
        # 1. Keyword Search
        tokenized_query = word_tokenize(query_str)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_n_bm25 = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:self.rerank_top_n]
        
        # 2. Vector Search
        query_embedding = self.embed_model.get_query_embedding(query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self.rerank_top_n
        )
        vector_results = self.vector_store.query(vector_store_query)
        
        # 3. RRF Fusion
        fused_scores = {}
        for rank, index in enumerate(top_n_bm25):
            node = self.nodes[index]
            fused_scores[node.node_id] = fused_scores.get(node.node_id, 0) + (1.0 / (rank + 60))
            
        if vector_results.ids:
            for rank, doc_id in enumerate(vector_results.ids):
                 fused_scores[doc_id] = fused_scores.get(doc_id, 0) + (1.0 / (rank + 60))
        
        # Get Top Candidates for Re-ranking
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)[:self.rerank_top_n]
        
        candidate_nodes = []
        for doc_id in sorted_ids:
            for node in self.nodes:
                if node.node_id == doc_id:
                    candidate_nodes.append(node)
                    break

        # --- STAGE 2: CROSS-ENCODER RERANKING ---
        if not candidate_nodes:
            return []

        # Prepare pairs: [("Query", "Document Text 1"), ("Query", "Document Text 2")...]
        pairs = [[query_str, node.get_content()] for node in candidate_nodes]
        
        # The AI reads the pairs and gives a relevance score
        rerank_scores = self.reranker.predict(pairs)
        
        # Combine Nodes with their new AI Scores
        final_results = []
        for node, score in zip(candidate_nodes, rerank_scores):
            final_results.append(NodeWithScore(node=node, score=float(score))) # Convert numpy float to python float
            
        # Sort by the NEW Cross-Encoder Score
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        # Return only the Top K (e.g., 5)
        return final_results[:self.top_k]