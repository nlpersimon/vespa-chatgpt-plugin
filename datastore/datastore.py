from abc import ABC, abstractmethod
from typing import List

from models.models import (
    Query,
    QueryResult,
    QueryWithEmbedding,
)
from services.embedding import get_embeddings


class DataStore(ABC):
    async def query(self, queries: List[Query]) -> List[QueryResult]:
        """
        Takes in a list of queries and filters and returns a list of query results with matching document chunks and scores.
        """
        # get a list of of just the queries from the Query list
        query_texts = [query.query for query in queries]
        query_embeddings = get_embeddings(query_texts)
        # hydrate the queries with embeddings
        queries_with_embeddings = [
            QueryWithEmbedding(**query.dict(), embedding=embedding)
            for query, embedding in zip(queries, query_embeddings)
        ]
        return await self._query(queries_with_embeddings)

    @abstractmethod
    async def _query(self, queries: List[QueryWithEmbedding]) -> List[QueryResult]:
        """
        Takes in a list of queries with embeddings and filters and returns a list of query results with matching document chunks and scores.
        """
        raise NotImplementedError