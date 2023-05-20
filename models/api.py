from models.models import (
    Document,
    Query,
    QueryResult,
)
from pydantic import BaseModel
from typing import List


class UpsertRequest(BaseModel):
    documents: List[Document]


class UpsertResponse(BaseModel):
    ids: List[str]


class QueryRequest(BaseModel):
    queries: List[Query]


class QueryResponse(BaseModel):
    results: List[QueryResult]
