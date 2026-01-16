from typing import List
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    
class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: bool