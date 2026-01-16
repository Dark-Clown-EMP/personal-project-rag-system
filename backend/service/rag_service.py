from fastapi import HTTPException
from backend.dtos.query_dto import QueryResponse

class RagService:
    @staticmethod
    def process_query(engine, query_text: str) -> QueryResponse:
        if not engine:
            raise HTTPException(status_code=503, detail="AI Engine is still loading...")

        try:
            # Query the LlamaIndex Engine
            response = engine.query(query_text)
            
            # Format Sources for the Frontend
            source_list = []
            if response.source_nodes:
                for node in response.source_nodes:
                    # Clean up newlines and truncate for display
                    text = node.node.get_content().replace("\n", " ")[:250]
                    score = float(node.score) if node.score else 0.0
                    source_list.append(f"[Score: {score:.2f}] {text}...")

            return QueryResponse(answer=str(response), sources=source_list)

        except Exception as e:
            print(f"🔴 [RAG ERROR] {str(e)}")
            raise HTTPException(status_code=500, detail="Internal AI processing error")