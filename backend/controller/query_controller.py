from fastapi import APIRouter, Request
from backend.dtos.query_dto import QueryRequest, QueryResponse, FeedbackRequest
from backend.service.rag_service import RagService
from backend.service.feedback_service import FeedbackService

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def ask_question(request: Request, query_data: QueryRequest):
    """
    Endpoint to process user queries using the RAG engine.
    """
    # Access the global engine from app.state
    engine = request.app.state.query_engine
    return RagService.process_query(engine, query_data.query)

@router.post("/feedback")
async def submit_feedback(feedback_data: FeedbackRequest):
    """
    Endpoint to save user feedback.
    """
    FeedbackService.save_feedback(feedback_data)
    return {"status": "success", "message": "Feedback saved"}