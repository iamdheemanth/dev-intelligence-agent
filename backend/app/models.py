from pydantic import BaseModel, Field
from typing import Optional, Dict

class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    top_k: int = Field(5, ge=1, le=20)

class Issue(BaseModel):
    title: str = Field(..., min_length=2, max_length=200)
    body: str = Field("", max_length=5000)

class CodeReviewRequest(BaseModel):
    code: str = Field(..., min_length=1)
    use_llm: bool = True

class DocstringRequest(BaseModel):
    code: str = Field(..., min_length=1)

class SummarizeRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=200)

class Recommendation(BaseModel):
    name: str
    reason: str
    score: float
    links: Optional[Dict] = None
