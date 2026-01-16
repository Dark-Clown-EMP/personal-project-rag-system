import nltk
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from nltk.tokenize import word_tokenize

# LlamaIndex Imports
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- NLTK SETUP ---
# Ensures tokenizers are ready when this class is imported
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_store, nodes, top_k=5, rerank_top_n=20):
        super().__init__()
        self.vector_store = vector_store
        self.nodes = nodes
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
        
        print("⏳ [INIT] Building BM25 Index (RAM)...")
        tokenized_corpus = [word_tokenize(node.get_content()) for node in nodes]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("✅ [INIT] BM25 Ready.")

        # Local Embedding Model
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        print("⏳ [INIT] Loading Cross-Encoder (MS-MARCO)...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("✅ [INIT] Cross-Encoder Ready.")

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query_str = query_bundle.query_str
        
        # 1. BM25 Search (Sparse)
        tokenized_query = word_tokenize(query_str)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_n_bm25 = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:self.rerank_top_n]
        
        # 2. Vector Search (Dense)
        query_embedding = self.embed_model.get_query_embedding(query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self.rerank_top_n
        )
        vector_results = self.vector_store.query(vector_store_query)
        
        # 3. Reciprocal Rank Fusion (RRF)
        fused_scores = {}
        for rank, index in enumerate(top_n_bm25):
            node = self.nodes[index]
            fused_scores[node.node_id] = fused_scores.get(node.node_id, 0) + (1.0 / (rank + 60))
            
        if vector_results.ids:
            for rank, doc_id in enumerate(vector_results.ids):
                 fused_scores[doc_id] = fused_scores.get(doc_id, 0) + (1.0 / (rank + 60))
        
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)[:self.rerank_top_n]
        candidate_nodes = [n for n in self.nodes if n.node_id in sorted_ids]

        if not candidate_nodes:
            return []

        # 4. Cross-Encoder Re-ranking
        pairs = [[query_str, node.get_content()] for node in candidate_nodes]
        rerank_scores = self.reranker.predict(pairs)
        
        final_results = []
        for node, score in zip(candidate_nodes, rerank_scores):
            final_results.append(NodeWithScore(node=node, score=float(score)))
            
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:self.top_k]