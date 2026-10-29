import numpy as np
from openai import OpenAI
import tiktoken

from .base import BaseReranker, register_reranker


@register_reranker("api")
class ApiReranker(BaseReranker):
    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        if not s1 or not s2:
            return np.zeros((len(s1), len(s2)))
        settings = self.config.reranker.api
        batch_size = int(settings.get("batch_size", 32))
        max_tokens = int(settings.get("max_tokens", 4096))
        if batch_size < 1 or max_tokens < 1:
            raise ValueError("Embedding batch_size and max_tokens must be positive")
        client = OpenAI(api_key=settings.key, base_url=settings.base_url)
        tokenizer = tiktoken.get_encoding("cl100k_base")
        texts = [tokenizer.decode(tokenizer.encode(s, disallowed_special=())[:max_tokens]) for s in s1 + s2]
        embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            response = client.embeddings.create(input=batch, model=settings.model)
            ordered = sorted(response.data, key=lambda row: row.index)
            if [row.index for row in ordered] != list(range(len(batch))):
                raise ValueError("Embedding API returned missing or duplicate indices")
            embeddings.extend(row.embedding for row in ordered)
        vectors = np.asarray(embeddings, dtype=float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / np.maximum(norms, 1e-12)
        return normalized[:len(s1)] @ normalized[len(s1):].T
