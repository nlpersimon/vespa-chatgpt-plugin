import os
from typing import Any, Dict, List, Optional
import pinecone
from tenacity import retry, wait_random_exponential, stop_after_attempt
import asyncio

from datastore.datastore import DataStore
from models.models import (
    DocumentChunkMetadata,
    DocumentChunkWithScore,
    DocumentMetadataFilter,
    QueryResult,
    QueryWithEmbedding,
    Source,
)
from services.date import to_unix_timestamp

# Read environment variables for Pinecone configuration
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
assert PINECONE_API_KEY is not None
assert PINECONE_ENVIRONMENT is not None
assert PINECONE_INDEX is not None

# Initialize Pinecone with the API key and environment
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Set the batch size for upserting vectors to Pinecone
UPSERT_BATCH_SIZE = 100


class PineconeDataStore(DataStore):
    def __init__(self):
        # Check if the index name is specified and exists in Pinecone
        if PINECONE_INDEX and PINECONE_INDEX not in pinecone.list_indexes():

            # Get all fields in the metadata object in a list
            fields_to_index = list(DocumentChunkMetadata.__fields__.keys())

            # Create a new index with the specified name, dimension, and metadata configuration
            try:
                print(
                    f"Creating index {PINECONE_INDEX} with metadata config {fields_to_index}"
                )
                pinecone.create_index(
                    PINECONE_INDEX,
                    dimension=1536,  # dimensionality of OpenAI ada v2 embeddings
                    metadata_config={"indexed": fields_to_index},
                )
                self.index = pinecone.Index(PINECONE_INDEX)
                print(f"Index {PINECONE_INDEX} created successfully")
            except Exception as e:
                print(f"Error creating index {PINECONE_INDEX}: {e}")
                raise e
        elif PINECONE_INDEX and PINECONE_INDEX in pinecone.list_indexes():
            # Connect to an existing index with the specified name
            try:
                print(f"Connecting to existing index {PINECONE_INDEX}")
                self.index = pinecone.Index(PINECONE_INDEX)
                print(f"Connected to index {PINECONE_INDEX} successfully")
            except Exception as e:
                print(f"Error connecting to index {PINECONE_INDEX}: {e}")
                raise e

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
    async def _query(
        self,
        queries: List[QueryWithEmbedding],
    ) -> List[QueryResult]:
        """
        Takes in a list of queries with embeddings and filters and returns a list of query results with matching document chunks and scores.
        """

        # Define a helper coroutine that performs a single query and returns a QueryResult
        async def _single_query(query: QueryWithEmbedding) -> QueryResult:
            print(f"Query: {query.query}")

            # Convert the metadata filter object to a dict with pinecone filter expressions
            pinecone_filter = self._get_pinecone_filter(query.filter)

            try:
                # Query the index with the query embedding, filter, and top_k
                query_response = self.index.query(
                    # namespace=namespace,
                    top_k=query.top_k,
                    vector=query.embedding,
                    filter=pinecone_filter,
                    include_metadata=True,
                )
            except Exception as e:
                print(f"Error querying index: {e}")
                raise e

            query_results: List[DocumentChunkWithScore] = []
            for result in query_response.matches:
                score = result.score
                metadata = result.metadata
                # Remove document id and text from metadata and store it in a new variable
                metadata_without_text = (
                    {key: value for key, value in metadata.items() if key != "text"}
                    if metadata
                    else None
                )

                # If the source is not a valid Source in the Source enum, set it to None
                if (
                    metadata_without_text
                    and "source" in metadata_without_text
                    and metadata_without_text["source"] not in Source.__members__
                ):
                    metadata_without_text["source"] = None

                # Create a document chunk with score object with the result data
                result = DocumentChunkWithScore(
                    id=result.id,
                    score=score,
                    text=metadata["text"] if metadata and "text" in metadata else None,
                    metadata=metadata_without_text,
                )
                query_results.append(result)
            return QueryResult(query=query.query, results=query_results)

        # Use asyncio.gather to run multiple _single_query coroutines concurrently and collect their results
        results: List[QueryResult] = await asyncio.gather(
            *[_single_query(query) for query in queries]
        )

        return results

    def _get_pinecone_filter(
        self, filter: Optional[DocumentMetadataFilter] = None
    ) -> Dict[str, Any]:
        if filter is None:
            return {}

        pinecone_filter = {}

        # For each field in the MetadataFilter, check if it has a value and add the corresponding pinecone filter expression
        # For start_date and end_date, uses the $gte and $lte operators respectively
        # For other fields, uses the $eq operator
        for field, value in filter.dict().items():
            if value is not None:
                if field == "start_date":
                    pinecone_filter["date"] = pinecone_filter.get("date", {})
                    pinecone_filter["date"]["$gte"] = to_unix_timestamp(value)
                elif field == "end_date":
                    pinecone_filter["date"] = pinecone_filter.get("date", {})
                    pinecone_filter["date"]["$lte"] = to_unix_timestamp(value)
                else:
                    pinecone_filter[field] = value

        return pinecone_filter
