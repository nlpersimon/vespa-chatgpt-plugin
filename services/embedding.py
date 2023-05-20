from typing import List
from tenacity import retry, wait_random_exponential, stop_after_attempt
from sentence_transformers import SentenceTransformer

# Instantiate a transformer model. You can replace 'all-MiniLM-L6-v2' with the model of your choice.
model = SentenceTransformer('all-MiniLM-L6-v2')


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_embeddings(texts: List[str]) -> List[List[float]]:
    embeddings = [e.tolist() for e in model.encode(texts)]
    return embeddings
