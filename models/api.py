from models.models import (
    Query,
    QueryResult,
)
from pydantic import BaseModel
from typing import List



class QueryRequest(BaseModel):
    queries: List[Query]


class QueryResponse(BaseModel):
    results: List[QueryResult]
