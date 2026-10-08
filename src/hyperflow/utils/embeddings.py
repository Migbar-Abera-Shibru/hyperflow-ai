"""
Embeddings module for HyperFlow AI.

Provides semantic embeddings for:
- Tool descriptions (for task-tool matching)
- Schema descriptions (for node anchoring)
- Dependency inference (semantic similarity between schemas)

Supports multiple backends:
- Sentence Transformers (local, free)
- OpenAI embeddings (higher quality, paid)
- Groq embeddings (free, fast)

Production considerations:
- Caching embeddings for performance
- Batch processing for efficiency
- Fallback to simpler similarity when embeddings unavailable
- Configurable embedding model

Usage:
    embedder = EmbeddingManager()
    
    # Single embedding
    vec = embedder.embed("Get user details")
    
    # Batch embedding
    vecs = embedder.embed_batch(["tool1 desc", "tool2 desc"])
    
    # Similarity
    sim = embedder.similarity("user_id", "recipient_id")
"""



from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """ configuration for embedding models."""
    model_type: str = "sentence_transformers" # sentence_transofmers, openai, groq
    model_name: str = "all_MiniLM-L6-v2"
    cache_size: int = 1000
    cache_dir: Optional[str] = None
    batch_size: int = 32
    device: str = "cpu" # cpu, cuda
    use_cache: bool = True

class EmbeddingCache:
    """
    LRU cache for embeddings.
    
    Stores embeddings in memory and optionally persists to disk.
    """
    def __init__(self, max_size: int = 10000, persist_dir: Optional[str] = None):
        self.max_size = max_size
        self.persist_dir = persist_dir
        self._cache: Dict[str, np.ndarray] = {}
        self._access_order: List[str] = []

        if persist_dir:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[np.ndarray]:
        """get embedding from cache."""
        if key in self._cache:
            # update access order
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]

        # try to load from disk
        if self.persist_dir:
            cache_file = Path(self.persist_dir) / f"{hashlib.md5(key.encode()).hexdigest()}.npy"

            if cache_file.exists():
                embedding = np.load(cache_file)
                self._cache[key] = embedding
                self._access_order.append(key)
                return embedding

        return None

